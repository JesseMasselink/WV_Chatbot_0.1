# Does not work
#from pandasai.llm.local_llm import LocalLLM

import pandasai as pai
from pandasai_litellm.litellm import LiteLLM


llm = LiteLLM(
    model="ollama/codellama:7b",
    api_key="82fe034cdb664ebb936d7121397f664d.lf_w5tGConKuske3_nfHXLDo"
)

pai.config.set({
    "llm": llm
})

df = pai.read_csv(r"/home/aiadmin/WasteVision/GAD2/designatedlocation_export.csv")

response = df.chat("How many times were containers changed or updated in total?")
print(response)