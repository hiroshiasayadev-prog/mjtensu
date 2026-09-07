from __future__ import annotations

import math
import unittest

from mldb.src.common.parameters import (
    PublicParameterDeclaration,
    is_public_parameter_value,
    resolve_public_parameters,
)


class CommonParametersTests(unittest.TestCase):
    def test_recursive_json_values_are_accepted(self) -> None:
        value = {
            "none": None,
            "bool": True,
            "str": "x",
            "int": 3,
            "float": 0.25,
            "list": [1, False, {"nested": [None, "ok"]}],
        }

        self.assertTrue(is_public_parameter_value(value))
        self.assertFalse(is_public_parameter_value({1: "bad"}))
        self.assertFalse(is_public_parameter_value((1, 2)))
        self.assertFalse(is_public_parameter_value(math.nan))
        self.assertFalse(is_public_parameter_value(math.inf))
        self.assertFalse(is_public_parameter_value(-math.inf))

        cyclic: list[object] = []
        cyclic.append(cyclic)
        self.assertFalse(is_public_parameter_value(cyclic))

    def test_bool_and_int_are_preserved_without_coercion(self) -> None:
        resolved = resolve_public_parameters(
            {
                "flag": PublicParameterDeclaration(default=False),
                "count": PublicParameterDeclaration(default=1),
            },
            {"flag": True, "count": 2},
        )

        self.assertIs(type(resolved["flag"]), bool)
        self.assertIs(type(resolved["count"]), int)

    def test_unknown_parameter_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_public_parameters(
                {"known": PublicParameterDeclaration(default=1)},
                {"unknown": 2},
            )

    def test_defaults_fill_complete_mapping_and_nested_values_are_copied(self) -> None:
        default_nested = {"items": [1, {"enabled": True}]}
        override_nested = ["a", {"n": 2}]
        declarations = {
            "defaulted": PublicParameterDeclaration(default=default_nested),
            "overridden": PublicParameterDeclaration(default=[]),
        }
        overrides = {"overridden": override_nested}

        resolved = resolve_public_parameters(declarations, overrides)

        self.assertEqual(
            {
                "defaulted": {"items": [1, {"enabled": True}]},
                "overridden": ["a", {"n": 2}],
            },
            resolved,
        )
        self.assertIsNot(default_nested, resolved["defaulted"])
        self.assertIsNot(default_nested["items"], resolved["defaulted"]["items"])
        self.assertIsNot(override_nested, resolved["overridden"])
        self.assertIsNot(override_nested[1], resolved["overridden"][1])

        default_nested["items"].append(99)
        override_nested.append("later")
        self.assertEqual([1, {"enabled": True}], resolved["defaulted"]["items"])
        self.assertEqual(["a", {"n": 2}], resolved["overridden"])

    def test_invalid_default_or_override_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_public_parameters(
                {"bad": PublicParameterDeclaration(default=float("nan"))},
                {},
            )
        with self.assertRaises(ValueError):
            resolve_public_parameters(
                {"bad": PublicParameterDeclaration(default=0.0)},
                {"bad": float("inf")},
            )


if __name__ == "__main__":
    unittest.main()
