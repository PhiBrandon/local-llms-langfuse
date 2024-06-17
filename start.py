import ollama
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from langfuse import Langfuse

load_dotenv()


class UserDetail(BaseModel):
    name: str
    age: int


langfuse = Langfuse()

trace_id = str(uuid.uuid4())
ollama_trace = langfuse.trace(id=trace_id, name="ollama_video_trace")


user_prompt = "Brandon is 33"
instruction_prompt = f"Extract name and age from the following text:\n{user_prompt}\nOutput valid JSON using the following schema but don't repeat the schema: {UserDetail.model_json_schema()}"
ollama_generation = ollama_trace.generation(
    name="qwen2-generation-youtube",
    input={
        "prompt": user_prompt,
        "instruction_prompt": instruction_prompt,
        "model_schema": UserDetail.model_json_schema(),
    },
    metadata={"version": 1}
)

output = ollama.generate("qwen2-7b:latest", prompt=instruction_prompt)

ollama_generation.end(
    output=output.get("response"),
)
ollama_trace.update(input=user_prompt, output=output)

# eval_count
# prompt_eval_count
