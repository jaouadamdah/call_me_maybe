import numpy as np
from typing import Any

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .NextTokenSelector import NextTokenSelector
from .state_generator import StateGenerator


class Generator:
    """
    Manages the token-by-token generation process with constrained decoding.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        states: StateGenerator,
        get_next_token: NextTokenSelector,
        vocab: dict[Any, Any],
        max_token: int | None = None,
    ) -> None:
        """Initializes the generator.

        Args:
            model (Small_LLM_Model): The LLM SDK.
            states (StateGenerator): The states handling schema rules.
            next_token_selector (NextTokenSelector): The token filtering logic.
            vocab (dict[str, int]): The vocabulary mapping strings to IDs.
            max_token (int | None): Max tokens to generate. Defaults to None.
        """

        self.model: Small_LLM_Model = model
        self.states: StateGenerator = states
        self.get_next_token: NextTokenSelector = get_next_token
        self.vocab: dict[int, str] = {id: token for token, id in vocab.items()}
        self.max_token = max_token

    def call(self, prompt: str) -> str:
        """Executes the constrained generation loop for a given prompt.

        Args:
            prompt (str): The initial string prompt.

        Returns:
            str: The generated text.
        """

        state_id = 0
        tokens = 0
        ids: list[int] = self.model.encode(prompt)[0].tolist()
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
            print(self.vocab[token_id], end="", flush=True)
            if self.max_token:
                tokens += 1
        print()
        if (
            not self.states.is_end(state_id)
            and self.max_token
            and tokens >= self.max_token
        ):
            print("[Warning] Max tokens reached!")

        res: str = self.model.decode(ids[generated:])
        return res
