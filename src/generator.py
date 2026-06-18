import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import BaseModel, model_validator, ConfigDict, Field

from .NextTokenSelector import NextTokenSelector
from .state_generator import StateGenerator


class Generator(BaseModel):
    """
    Manages the token-by-token generation process with constrained decoding.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    states: StateGenerator
    get_next_token: NextTokenSelector
    raw_vocab: dict[str, int]
    max_token: int | None = None
    vocab: dict[int, str] = Field(default_factory=dict, init=False)

    @model_validator(mode="after")
    def build_vocab(self) -> "Generator":
        """
        Builds a reverse vocabulary mapping from token IDs to tokens.
        """

        self.vocab = {id: token for token, id in self.raw_vocab.items()}
        return self

    def call(self, prompt: str) -> str:
        """Executes the constrained generation loop for a given prompt.

        Args:
            prompt (str): The initial string prompt.

        Returns:
            str: The generated text.
        """

        state_id = 0
        tokens = 0
        ids: list[int] = self.encode(prompt)
        generated: int = len(ids)

        while not self.states.is_end(state_id) and (
            not self.max_token or tokens < self.max_token
        ):
            logits = np.array(self.model.get_logits_from_input_ids(ids))
            token_id, state_id = self.get_next_token.run(
                state_id,
                logits,
            )
            ids.append(token_id)

            self.show_token(token_id)

            if self.max_token:
                tokens += 1

        print()

        if (
            not self.states.is_end(state_id)
            and self.max_token
            and tokens >= self.max_token
        ):
            print("[Warning] Max tokens reached!")

        return self.decode(ids[generated:])

    def encode(self, text: str) -> list[int]:
        """
        Encodes the given text into a list of token IDs.
        Args:
            text (str): The text to encode.
        Returns:
            list[int]: A list of token IDs corresponding to the input text.
        """

        return self.model.encode(text)[0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Args:
            token_ids (list[int]): A list of token IDs to decode.
        Returns:
            str: The decoded string corresponding to the input token IDs.
        """

        return self.model.decode(token_ids)

    def show_token(self, token_id: int) -> None:
        """
        Displays the token corresponding to the given token ID.
        Args:
            token_id (int): The token ID to display.
        """

        print(self.vocab[token_id], end="", flush=True)
