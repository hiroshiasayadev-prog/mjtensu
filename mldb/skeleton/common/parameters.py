"""Shared MLDB public-parameter signatures.

This skeleton fixes the common parameter boundary shared by Train Protocol,
Evaluation Protocol, direct Run launch, and Study materialization. It defines
only the JSON-compatible public value domain and default/override resolution.
Study grid semantics, training seed, protocol-specific constraints, YAML
parsing, coercion, and generic range/type schemas are intentionally outside
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias, TypeGuard


PublicParameterValue: TypeAlias = (
    None
    | bool
    | str
    | int
    | float
    | list["PublicParameterValue"]
    | dict[str, "PublicParameterValue"]
)
"""One MLDB v1 public parameter value.

The recursive domain is the JSON-compatible data model: null, bool, string,
integer, finite float, arrays, and objects with string keys. ``float`` in this
static alias is narrower at runtime: NaN and positive/negative infinity are
invalid and must be rejected by :func:`is_public_parameter_value` and parameter
resolution.

Built-in ``list`` and ``dict`` represent JSON arrays and objects. Arbitrary
Python objects, custom scalar coercions, non-string object keys, and
implementation-specific tagged values are not part of this contract.
"""


@dataclass(frozen=True, slots=True)
class PublicParameterDeclaration:
    """Normalized common portion of one public parameter declaration.

    ``default`` is required. An authored declaration without ``default`` is
    invalid and must be rejected before it can be represented by this valid
    declaration type.

    Protocol-format metadata such as ``description``, ``type``, ``minimum``,
    ``maximum``, or ``suggested`` is deliberately absent because MLDB v1 gives
    those fields no common resolution semantics.
    """

    default: PublicParameterValue


PublicParameterDeclarations: TypeAlias = Mapping[str, PublicParameterDeclaration]
"""Published parameter declarations keyed by protocol-local public name."""

PublicParameterOverrides: TypeAlias = Mapping[str, PublicParameterValue]
"""Caller-supplied values for a subset of published parameter names."""

ResolvedPublicParameters: TypeAlias = Mapping[str, PublicParameterValue]
"""Complete resolved mapping containing every published parameter exactly once.

Recursive JSON arrays/objects retain ``list``/``dict`` representation so
resolution does not coerce the persisted public value domain into non-JSON
wrapper types. The concrete mapping implementation is not fixed here.
"""


def is_public_parameter_value(value: object) -> TypeGuard[PublicParameterValue]:
    """Return whether ``value`` belongs to the MLDB public-parameter domain.

    Validation is recursive. ``bool`` is a valid public parameter type even
    though Python implements ``bool`` as an ``int`` subclass; implementations
    must preserve the authored boolean/integer distinction rather than coerce
    between them. Integers have no common MLDB numeric bound. Floats are valid
    only when finite.

    Arrays must contain only valid public values. Objects must have string keys
    and valid public values. No string/number/bool coercion is permitted.
    """

    ...


def resolve_public_parameters(
    declarations: PublicParameterDeclarations,
    overrides: PublicParameterOverrides,
) -> ResolvedPublicParameters:
    """Resolve protocol defaults plus caller overrides into one complete mapping.

    For every published key, the caller value wins when present; otherwise the
    declaration default is used. The result contains exactly the published keys
    and no protocol declaration metadata.

    Resolution rejects an unknown override key and any default or override value
    outside :class:`PublicParameterValue`, including NaN or infinity. It performs
    no implicit coercion and applies no generic range, enum, model-family, or
    protocol-specific semantic validation.

    Valid declarations always contain ``default`` by construction. Raw protocol
    parsing/format validation must therefore reject a declaration that omits
    ``default`` before producing ``PublicParameterDeclaration`` objects.

    Recursive values remain ordinary JSON-compatible list/dict values rather
    than being coerced to non-JSON immutable wrappers. The concrete mapping
    class and copy/mutation strategy are implementation details.

    Failure is exception-based: invalid input produces no partial mapping and no
    result/report union. The concrete exception type is not fixed by this common
    parameter signature; application callers may translate resolution failure at
    their own validation/error boundary.
    """

    ...
