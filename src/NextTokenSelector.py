import numpy as np

from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Any

from .state_generator import StateGenerator


class NextTokenSelector(BaseModel):
    """Filters logits based on states to enforce valid token generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_vocab: dict[str, int]
    states: StateGenerator
    cache: dict[int, tuple[int, np.ndarray[Any, Any], dict[int, int]]] = Field(
        default_factory=dict, init=False
    )
    vocab: dict[str, Any] = Field(
        default_factory=lambda: {"any": set()},
        init=False,
    )
    logits_len: int | None = Field(default=None, init=False)
    logits_ids: set[int] = Field(default_factory=set, init=False)

    @model_validator(mode="after")
    def build_vocab(self) -> "NextTokenSelector":

        def check_token(token: str) -> bool:
            for char in token:
                if char == '"' or char == "\\":
                    return False
            return True

        for token, token_id in self.raw_vocab.items():
            self.vocab.setdefault(token[0], {})[token] = token_id
            if check_token(token):
                self.vocab["any"].add(token_id)

        return self

    def run(self, curr_state_id: int, logits: np.ndarray[Any, Any]
            ) -> tuple[int, int]:
        """Modifies logits based on the current state and selects the next \
            token.

        Args:
            curr_state_id (int): The current state.
            logits (np.ndarray): The raw probability from the LLM.

        Returns:
            tuple[int, int]: The selected token ID and the next state ID.
        """

        cached_data = self.cache.get(curr_state_id)

        if cached_data is not None:
            state_type, token_ids, allowed_tokens_map = cached_data

            if state_type:
                max_id = logits[token_ids].argmax()
                next_token = token_ids[max_id]
            else:
                logits[token_ids] = float("-inf")
                next_token = logits.argmax()

            return next_token, allowed_tokens_map[next_token]

        allowed_tokens = self.get_allowed_tokens(curr_state_id)

        if self.logits_len is None:
            self.logits_len = len(logits)
            self.logits_ids = set(range(self.logits_len))

        if self.states.is_any(curr_state_id):
            # cache invalid tokens id
            invalid_ids = np.array(list(self.logits_ids - set(allowed_tokens)))
            self.cache[curr_state_id] = (0, invalid_ids, allowed_tokens)
        else:
            # cache valid tokens id
            valid_ids = np.array(list(allowed_tokens))
            self.cache[curr_state_id] = (1, valid_ids, allowed_tokens)

        return self.run(curr_state_id, logits)

    def get_allowed_tokens(
        self,
        curr_state_id: int,
    ) -> dict[int, int]:
        """Calculates which tokens are valid transitions from the current\
              state.

        Args:
            curr_state_id (int): The current state.

        Returns:
            dict[int, int]: A mapping of allowed token IDs to their target \
                state IDs.
        """

        tokens: dict[int, int] = {}

        if self.states.is_any(curr_state_id):
            any_set: set[int] = self.vocab["any"]
            tokens = dict.fromkeys(any_set, curr_state_id)

        for state_token in self.states.states.get(curr_state_id, {}):
            for token, token_id in self.vocab.get(state_token, {}).items():
                is_allowed = True
                next_state_id: int | None = self.states.next_state(
                    curr_state_id, token[0]
                )

                if next_state_id is None:
                    raise ValueError(
                        "Error occurred while getting allowed tokens, "
                        f"invalid state token: {state_token}"
                    )

                for i in range(1, len(token)):
                    if self.states.is_end(next_state_id):
                        is_allowed = False
                        break
                    new_state_id: int | None = self.states.next_state(
                        next_state_id,
                        token[i],
                    )
                    if new_state_id is None:
                        if not self.states.is_any(next_state_id):
                            is_allowed = False
                            break
                    else:
                        next_state_id = new_state_id

                if is_allowed:
                    tokens[token_id] = next_state_id
        if not tokens:
            raise ValueError(
                "Error: Could not find any allowed tokens,"
                f" current state id: {curr_state_id}"
            )
        return tokens
