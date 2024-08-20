"""This module is an entry point for the application to launch user interface."""

import os
import sys
import json
import pandas as pd
import gradio as gr
from secret_manager import SecretManager

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason, GenerationConfig
import vertexai.preview.generative_models as generative_models

from langchain_community.document_loaders import UnstructuredExcelLoader

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Set VARs from ENV Vars
os.environ.get('USER')
PROJECT_ID=os.environ.get('PROJECT_ID', 'nuttee-lab-00')
LOCATION=os.environ.get('LOCATION', 'us-central1')
REGION=os.environ.get('REGION', 'us-central1-b')
SEARCH_ENGINE_ID=os.environ.get('SEARCH_ENGINE_ID', 'set_taxonomy')
DATA_STORE_LOCATION=os.environ.get('DATA_STORE_LOCATION', 'global')
MAX_DOCUMENTS=os.environ.get('MAX_DOCUMENTS', "5")
ENGINE_DATA_TYPE=os.environ.get('ENGINE_DATA_TYPE', "1")

DATA_STORE_ID=f"{SEARCH_ENGINE_ID}_{PROJECT_ID}_01"
SEARCH_APP_ID=os.environ.get('SEARCH_APP_ID', 'set_taxonomy_app')

# Initialize Langfuse handler
from langfuse.decorators import observe, langfuse_context
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)

# Vertex AI Init
vertexai.init(project=PROJECT_ID, location=LOCATION)

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

model = GenerativeModel(
    "gemini-1.5-pro-001"
)

# System Instructions and Prompt Templates

textsi_taxonomy_mapping = """You are a helpful assistant to review the finacial reports and compare to taxonomy list.
Ensure your answers are complete, unless the user requests a more concise approach.
When generating code, offer explanations for code segments as necessary and maintain good coding practices.
When presented with inquiries seeking information, provide answers that reflect a deep understanding of the field, guaranteeing their correctness.
For any non-english queries, respond in the same language as the prompt unless otherwise specified by the user.
For prompts involving reasoning, provide a clear explanation of each step in the reasoning process before presenting the final answer."""

taxonomy_json_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query submitted by the user."
        },
        "set_code": {
            "type": "string",
            "description": "The SET Code corresponding to the identified item in the taxonomy."
        },
        "item_in_taxonomy": {
            "type": "string",
            "description": "The name of the item in the taxonomy that matches the query. Use NULL, if can not answer, or can not match"
        },
        "reasoning": {
            "type": "string",
            "description": "The reasoning behind the selection of the SET code and item in the taxonomy."
        }
    },
    "required": [
        "query",
        "set_code",
        "item_in_taxonomy",
        "reasoning"
    ]
}

# XLSX Document Loader Function
@observe()
def xlsx_document_loader(
    file_path: str,
):
    loader = UnstructuredExcelLoader(file_path, mode="single")
    docs = loader.load()

    print("XLSX Document parsed successfully.")

    return docs

@observe(as_type="generation")
def tables_extraction(
    parsed_xlsx: str,
):
    tables_schema = {
      "type": "ARRAY",
      "items": {
        "description": "Represents financial statement data.",
        "type": "OBJECT",
        "required": ["table_number", "table", "date", "headers", "currency_unit", "data"],
        "properties": {
          "table_number": {
            "description": "Sequence number of the table appeared from the source.",
            "type": "STRING"
          },
          "table": {
            "description": "The name of the financial statement.",
            "type": "STRING"
          },
          "date": {
            "description": "The date on which the financial statement is prepared.",
            "type": "STRING"
          },
          "headers": {
            "description": """Column headers for the financial data.
              Example: 
              - [\"งบการเงินที่แสดง เงินลงทุนตามวิธีส่วนได้เสีย พ.ศ. 2566 (บาท)\",\"งบการเงินที่แสดง เงินลงทุนตามวิธีส่วนได้เสีย พ.ศ. 2565 (บาท)\",\"งบการเงินที่แสดง งบการเงินเฉพาะกิจการ พ.ศ. 2566 (บาท)\",\"งบการเงินที่แสดง งบการเงินเฉพาะกิจการ พ.ศ. 2565 (บาท)\"]
              """,
            "type": "ARRAY",
            "items": {
              "type": "STRING"
            }
          },
          "currency_unit": {
            "description": "The currency in which the financial data is expressed.",
            "type": "STRING"
          },
        }
      }
    }

    prompt_tables_extraction = f"""On a given financial report data extracted from an excel file as CONTEXT, extract the data by follow instructions.

<INSTRUCTIONS>
- You must use data only from CONTEXT
- Use original language from CONTEXT
- Extract all tables information to JSON format
</INSTRUCTIONS>

<CONTEXT>
{parsed_xlsx}
</CONTEXT>

Ouput in JSON:
"""

    input = [prompt_tables_extraction]

    response = model.generate_content(
        input,
        generation_config=GenerationConfig(
            max_output_tokens=8192,
            temperature=0.2,
            top_p=0.95,
            response_mime_type="application/json",
            response_schema=tables_schema,
            frequency_penalty=0.5,
        ),
        safety_settings=safety_settings,
        stream=False,
    )

    tables_json = json.loads(response.text)

    print("Found tables:")
    for table in tables_json:
        print(table.get("table_number"))
        print(table.get("table"))
        print(table.get("date"))
        print(table.get("headers"))
        print(table.get("currency_unit"))

    # Update langfuse observation
    input_count = model.count_tokens(str(input))
    output_count = model.count_tokens(response.candidates[0].content)
    langfuse_context.update_current_observation(
        input=str(input),
        output=str(response.text),
        usage={
            # usage
            "input": input_count.total_billable_characters,
            "output": output_count.total_billable_characters,
            #"total": ,  # if not set, it is derived from input + output
            "unit": "CHARACTERS",  # any of: "TOKENS", "CHARACTERS", "MILLISECONDS", "SECONDS", "IMAGES"
        },
        model_parameters={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        },
        model="gemini-1.5-pro",
    )
    
    return json.loads(response.text)

@observe(as_type="generation")
def extract_table_data(
    table_info: str,
    parsed_xlsx: str,
):
    response_schema = {
      "type": "ARRAY",
      "items": {
        "description": "Represents financial statement data.",
        "type": "OBJECT",
        "required": ["table", "date", "headers", "currency_unit", "data"],
        "properties": {
          "table": {
            "description": "The name of the financial statement.",
            "type": "STRING"
          },
          "date": {
            "description": "The date on which the financial statement is prepared.",
            "type": "STRING"
          },
          "headers": {
            "description": "Column headers for the financial data.",
            "type": "ARRAY",
            "items": {
              "type": "STRING"
            }
          },
          "currency_unit": {
            "description": "The currency in which the financial data is expressed.",
            "type": "STRING"
          },
          "data": {
            "description": "The financial data itself.",
            "type": "ARRAY",
            "items": {
              "type": "OBJECT",
              "properties": {
                "category": {
                  "description": "The category or account name.",
                  "type": "STRING"
                },
                "items": {
                  "description": "Line items within a category.",
                  "type": "ARRAY",
                  "items": {
                    "type": "OBJECT",
                    "required": ["name", "values"],
                    "properties": {
                      "name": {
                        "description": "The name of the line item.",
                        "type": "STRING"
                      },
                      "note": {
                        "description": "Reference to a footnote for additional information.",
                        "type": "STRING"
                      },
                      "values": {
                        "description": "The financial values for each header.",
                        "type": "ARRAY",
                        "items": {
                          "type": "NUMBER"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    prompt_with_schema = f"""On a given financial report data extracted from an excel file as CONTEXT, extract the data by follow instructions.

<INSTRUCTIONS>
- You must use data only from CONTEXT
- Use original language from CONTEXT
- Extract the table {table_info} data to JSON format
</INSTRUCTIONS>

<CONTEXT>
{parsed_xlsx}
</CONTEXT>

Ouput in JSON:
"""

    input = [prompt_with_schema]

    response = model.generate_content(
        input,
        generation_config=GenerationConfig(
            max_output_tokens=8192,
            temperature=0.2,
            top_p=0.95,
            response_mime_type="application/json",
            response_schema=response_schema
        ),
        safety_settings={
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
        stream=False,
    )

    # Update langfuse observation
    input_count = model.count_tokens(str(input))
    output_count = model.count_tokens(response.candidates[0].content)
    langfuse_context.update_current_observation(
        input=str(input),
        output=str(response.text),
        usage={
            # usage
            "input": input_count.total_billable_characters,
            "output": output_count.total_billable_characters,
            #"total": ,  # if not set, it is derived from input + output
            "unit": "CHARACTERS",  # any of: "TOKENS", "CHARACTERS", "MILLISECONDS", "SECONDS", "IMAGES"
        },
        model_parameters={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        },
        model="gemini-1.5-pro",
    )

    return response

def response_info(response):
    # Response tokens count
    usage_metadata = response.usage_metadata
    print(f"Prompt Token Count: {usage_metadata.prompt_token_count}")
    print(f"Candidates Token Count: {usage_metadata.candidates_token_count}")
    print(f"Total Token Count: {usage_metadata.total_token_count}")

    print(response.candidates[0])

    print(response.text)

def llm_parser_xlsx_to_json(
    tables_json: list,
    parsed_xlsx: str,
):
    data_json = []

    for table in tables_json:
        print(f"Parsing table: {table.get('table_number')}: {table.get('table')}")
        response = extract_table_data(
            table_info=table,
            parsed_xlsx=parsed_xlsx,
        )
        data_json.append(json.loads(response.text))

    display_json = [ [x['table'], y['category'], z['name'], z['values']] for x in [ x[0] for x in data_json ] for y in x['data'] for z in y['items'] ]

    return display_json

@observe()
def xlsx_extraction_run(
    file_path: str,
):
    # Parse XLSX to Document
    doc = xlsx_document_loader(file_path)

    # Extract Tables list information
    tables_json = tables_extraction(
        parsed_xlsx=doc[0].page_content,
    )

    # Extract Tables data
    data_json = llm_parser_xlsx_to_json(
        tables_json=tables_json,
        parsed_xlsx=doc[0].page_content,
    )

    return data_json

@observe(as_type="generation")
def generate(
    input,
    generation_config: dict = None,
    safety_settings: dict = None,
    system_instruction: str = None,
    model: str = "gemini-1.5-pro-001",
    metadata: str = "NULL",
):
    model = GenerativeModel(
        model,
        system_instruction=[system_instruction]
    )
    response = model.generate_content(
        input,
        generation_config=GenerationConfig(
            max_output_tokens=8192,
            temperature=0.2,
            top_p=0.95,
            response_mime_type="application/json",
            response_schema=taxonomy_json_schema
        ),
        safety_settings=safety_settings,
        stream=False,
    )

    #print(response.text, end="")

    # Update langfuse observation
    input_count = model.count_tokens(str(input))
    output_count = model.count_tokens(response.candidates[0].content)
    langfuse_context.update_current_observation(
        input=str(input)+('|')+metadata,
        output=str(response.text),
        usage={
            # usage
            "input": input_count.total_billable_characters,
            "output": output_count.total_billable_characters,
            #"total": ,  # if not set, it is derived from input + output
            "unit": "CHARACTERS",  # any of: "TOKENS", "CHARACTERS", "MILLISECONDS", "SECONDS", "IMAGES"
        },
        model_parameters={
            "temperature": 0.2,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        },
        model="gemini-1.5-pro",
    )

    return response

@observe()
def search_structured_datastore(query: str) -> str:
    """Retrive the information of Taxomony list that help to identify SET Code from input query"""
    from langchain_community.retrievers import (
        GoogleVertexAISearchRetriever,
    )
    import json

    retriever = GoogleVertexAISearchRetriever(
        project_id=PROJECT_ID,
        search_engine_id=SEARCH_APP_ID,
        location_id=DATA_STORE_LOCATION,
        engine_data_type=1,
        max_documents=10,
    )

    results = retriever.invoke(query)

    docs = ""

    if len(results) == 0:
        return "Taxonomy Information Not Found. Please apologize and say I could not answer"
    for result in results:
        d = json.loads(result.page_content)
        docs += f"{d}\n"
        #docs += f"SET Code: {d.get('set_code')}\nItem in taxonomy: {d.get('item_in_taxonomy')}\nMeaning of taxonomy: {d.get('meaning_of_taxonomy')}\n\n"

    return docs

@observe()
def taxonomy_mapping(
    query: str,
    model: str = "gemini-1.5-pro-001",
    metadata: str = "NULL",
) -> str:
    """Retrive the information of Taxomony list that help to identify SET Code from input query"""

    # Search with query in Vertex AI Structured Datastore
    search_result = search_structured_datastore(query=query)

    # Prepare input for Gemini
    input = [
        """From the financial report line item name as INPUT.
Use the Taxonomy list in CONTEXT to suggest to correct SET Code and Item in Taxonomy Name""",
        f"""<CONTEXT>
{search_result}
</CONTEXT>""",
        f"INPUT: {query}",
        "OUTPUT IN JSON (query, set_code, item_in_taxonomy) and Reasoning:"
    ]

    # Generate response from Gemini
    response = generate(
        input,
        generation_config,
        safety_settings,
        textsi_taxonomy_mapping,
        model,
        metadata
    )

    return search_result, response.text

@observe()
def taxonomy_mapping_run(
    data_json: list,
    model: str = "gemini-1.5-pro-001",
):
    data_json_with_taxonomy = []

    for data in data_json:
        # Set Metadata to Table name, Category, and Number values
        metadata = data[0]+"-"+data[1]+"-"+str(data[3])
        # Gemini + Vertex AI Search Datastore Retriver
        predicted_taxonomy = taxonomy_mapping(query=data[2], model=model, metadata=metadata)

        # Parse result string to JSON
        data_json = json.loads(predicted_taxonomy[1])

        # Append to list
        data_json_with_taxonomy.append(
            {
                "query": data_json.get('query'),
                "set_code": data_json.get('set_code'),
                "item_in_taxonomy": data_json.get('item_in_taxonomy'),
                "reasoning": data_json.get('reasoning'),
                "table_name": data[0],
                "category": data[1],
                "number_values": str(data[3]),
            }
        )
        print("Input Query: " + data[2] + "\nPredicted: " + predicted_taxonomy[1])

    return data_json_with_taxonomy

@observe()
def blob_download(
    gcs_uri: str, # example: gs://nuttee-lab-00-genai/taxonomy/F3_BYD_Q2Y66.XLSX
):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    bucket_name = gcs_uri.split('/')[2]
    # The ID of your GCS object
    source_blob_name = '/'.join(gcs_uri.split('/')[3:])
    # The path to which the file should be downloaded
    destination_file_name = gcs_uri.split('/')[-1]

    from google.cloud import storage
    
    storage = storage.Client()
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    return destination_file_name

@observe(capture_input=False, capture_output=False)
def app_run(
    gcs_uri: str,
    model: str = "gemini-1.5-pro-001",
):
    # Download file from GCS
    file_path = blob_download(
        gcs_uri=gcs_uri,
    )

    # Extract XLSX file with Langchain Unstructured.io
    data_json = xlsx_extraction_run(
        file_path=file_path,
    )

    # Gemini run get tables information and number of tables, then run taxonomy mapping
    data_json_with_taxonomy = taxonomy_mapping_run(
        data_json=data_json,
        model=model,
    )

    langfuse_context.update_current_observation(
        input=gcs_uri, # any serializable object
        output=str(data_json_with_taxonomy), # any serializable object
    )

    return pd.DataFrame(data_json_with_taxonomy)

if __name__ == "__main__":
    with gr.Blocks() as app:
        input_text = gr.Textbox(label="XLSX File (GCS URI)")
        
        with gr.Row():
            output_df = gr.DataFrame(label="Output DataFrame")

        submit_btn = gr.Button("Map Taxonomy")
        submit_btn.click(
            fn=app_run,
            inputs=[input_text],
            outputs=[output_df],
        )
    
    app.launch(server_name="0.0.0.0", server_port=8080)