from typing import List

from fastapi import FastAPI
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

app = FastAPI()
llm = ChatOpenAI(temperature=0.0)


class Numbers(BaseModel):
    numbers: List[float]


reformat_query = """
Reformat the data in the request payload into the 
schema specified. The schema is compatible with
Pydantic, JSON Schema Core, JSON Schema Validation and OpenAPI.

Output the new request payload in a JSON format. Only output
the JSON by itself.

schema:
```
{schema}
```

request payload:
```
{payload}
```
"""
reformat_prompt = ChatPromptTemplate.from_template(reformat_query)
payload_chain = LLMChain(llm=llm, prompt=reformat_prompt)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors by attempting to reformat the request payload.

    Args:
        request (_type_): Request
        exc (_type_): Exception

    Returns:
        _type_: Response to caller.
    """
    try:
        parser = PydanticOutputParser(pydantic_object=Numbers)
        output = await payload_chain.arun(
            schema=Numbers.schema_json(), payload=exc.body
        )
        fixed_request = parser.parse(output)
        return await request.scope["endpoint"](fixed_request)
    except Exception:
        return await request_validation_exception_handler(request, exc)


@app.post("/sum/")
async def sum_payload(nums: Numbers) -> JSONResponse:
    """Sum a list of numbers.

    Args:
        nums (Numbers): Numbers to sum.

    Returns:
        JSONResponse: Response to caller with sum.
    """
    result = sum(nums.numbers)
    return JSONResponse(content={"sum": result})
