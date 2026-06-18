from llm_sdk import Small_LLM_Model


def encode(text: str): -> list[int]:
    start = 0
    end = len(text)

    while start != end:
        if self.raw_vocab.get(text[start:end]):
            start = end
            end = len(text)
        else:
            end -= 1

    return 
