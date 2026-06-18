from pydantic import BaseModel, model_validator, Field
from typing import Any


class StateGenerator(BaseModel):
    """Constructs and manages the states for JSON schema constraints."""

    tools: dict[str, Any]
    states: dict[int, dict[str, int]] = Field(default_factory=dict, init=False)
    next_state_id: int = Field(default=0, init=False)
    any_states: set[int] = Field(default_factory=set, init=False)
    end_states: set[int] = Field(default_factory=set, init=False)

    @model_validator(mode="after")
    def build_states(self) -> "StateGenerator":
        """Iterates over the tools schema to construct the states."""
        root_branch = self._build_sequence('{"name":"')

        for fn_name, data in self.tools.items():
            params = data["parameters"]

            last_state = self._build_sequence(
                f'{fn_name}","parameters":{{',
                root_branch,
            )

            length = len(params)
            for i, (arg_name, arg_type) in enumerate(params.items()):
                sep = "," if i < length - 1 else "}"

                last_state = self._build_sequence(f'"{arg_name}":', last_state)

                if arg_type == "number" or arg_type == "float":
                    last_state = self._build_number(last_state, sep)
                elif arg_type == "integer":
                    last_state = self._build_integer(last_state, sep)
                elif arg_type == "boolean":
                    last_state = self._build_boolean(last_state, sep)
                else:
                    last_state = self._build_string(last_state, sep)

            suffix = "}" if length else "}}"
            last_state = self._build_sequence(suffix, curr=last_state)

            self.end_states.add(last_state)
        return self

    def _add_state(self) -> int:
        """Creates a new empty state and returns its ID."""

        curr_state_id = self.next_state_id
        self.next_state_id += 1
        self.states.setdefault(curr_state_id, {})
        return curr_state_id

    def _add_transition(self, src: int, chars: str, dst: int) -> None:
        """
        Adds transitions for single or multiple characters from one state.
        """

        temp = self.states.setdefault(src, {})
        for char in chars:
            temp[char] = dst

    def _build_sequence(
        self, seq: str, curr: int | None = None, target: int | None = None
    ) -> int:
        """Builds a sequence of states for a specific string.

        Args:
            seq (str): The string sequence.
            curr (int | None, optional): The starting state ID.
            target (int | None, optional): The ending state ID.

        Returns:
            int: The ID of the final state in the sequence.
        """

        if curr is None:
            curr = self.next_state_id
            self.next_state_id += 1

        length = len(seq)
        for i, char in enumerate(seq):
            state = self.states.setdefault(curr, {})

            if char in state:
                curr = state[char]
            else:
                if target is not None and i == length - 1:
                    next_state = target
                else:
                    next_state = self._add_state()

                state[char] = next_state
                curr = next_state

        return curr

    def _build_string(self, prev_state: int, sep: str) -> int:
        """Builds states for string value."""

        loop_state = self._add_state()

        self._add_transition(prev_state, '"', loop_state)

        escaped_char_state = self._add_state()
        s_white_space = self._add_state()
        next_target = self._add_state()
        s_end = self._add_state()

        self.states.setdefault(loop_state, {}).update(
            {
                '"': s_end,
                "\\": escaped_char_state,
            }
        )
        self.any_states.add(loop_state)
        self._add_transition(escaped_char_state, '"\\/bfnrt', loop_state)

        self._add_transition(s_end, " ", s_white_space)
        self._add_transition(s_end, sep, next_target)
        self._add_transition(s_white_space, sep, next_target)

        return next_target

    def _build_boolean(self, prev_state: int, sep: str) -> int:
        """Builds states for boolean value."""

        s_white_space = self._add_state()
        s_end = self._add_state()
        next_target = self._add_state()

        self._build_sequence("true", prev_state, target=s_end)
        self._build_sequence("false", prev_state, target=s_end)

        self._add_transition(s_end, " ", s_white_space)
        self._add_transition(s_end, sep, next_target)
        self._add_transition(s_white_space, sep, next_target)

        return next_target

    def _build_number(self, prev_state: int, sep: str) -> int:
        """Builds states for number/float value"""

        s_sign = self._add_state()
        s_zero = self._add_state()
        s_integer = self._add_state()
        s_dot = self._add_state()
        s_fraction = self._add_state()
        s_white_space = self._add_state()
        next_target = self._add_state()

        self._add_transition(prev_state, "-", s_sign)
        self._add_transition(prev_state, "0", s_zero)
        self._add_transition(prev_state, "123456789", s_integer)

        self._add_transition(s_sign, "0", s_zero)
        self._add_transition(s_sign, "123456789", s_integer)
        self._add_transition(s_integer, "0123456789", s_integer)
        self._add_transition(s_integer, ".", s_dot)
        self._add_transition(s_zero, ".", s_dot)
        self._add_transition(s_dot, "0123456789", s_fraction)
        self._add_transition(s_fraction, "0123456789", s_fraction)

        for state in [s_zero, s_integer, s_fraction]:
            self._add_transition(state, " ", s_white_space)
            self._add_transition(state, sep, next_target)

        # self._add_transition(s_fraction, " ", s_white_space)
        # self._add_transition(s_fraction, sep, next_target)
        self._add_transition(s_white_space, sep, next_target)

        return next_target

    def _build_integer(self, prev_state: int, sep: str) -> int:
        """Builds states for integer value"""

        s_sign = self._add_state()
        s_zero = self._add_state()
        s_integer = self._add_state()

        self._add_transition(prev_state, "-", s_sign)
        self._add_transition(prev_state, "0", s_zero)
        self._add_transition(prev_state, "123456789", s_integer)

        self._add_transition(s_sign, "0", s_zero)
        self._add_transition(s_sign, "123456789", s_integer)
        self._add_transition(s_integer, "0123456789", s_integer)

        s_white_space = self._add_state()
        next_target = self._add_state()
        for state in [s_zero, s_integer]:
            self._add_transition(state, " ", s_white_space)
            self._add_transition(state, sep, next_target)
        self._add_transition(s_white_space, sep, next_target)

        return next_target

    def next_state(
        self,
        cur_state_id: int,
        token: str,
    ) -> int | None:
        """Retrieves the next state ID based on the current state and token.

        Args:
            cur_state_id (int): The current state ID.
            token (str): The character transition.

        Returns:
            int | None: The next state ID, or None if the transition is
                invalid.
        """

        return self.states.get(cur_state_id, {}).get(token)

    def is_any(self, state_id: int) -> bool:
        """Checks if the given state allows any string character."""

        if state_id in self.any_states:
            return True
        return False

    def is_end(self, state_id: int) -> bool:
        """Checks if the given state is a valid end state for the JSON \
            structure."""

        if state_id in self.end_states:
            return True
        return False
