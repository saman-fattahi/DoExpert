"""doexpert.doe — design-of-experiments utilities.

Currently exposes the Taguchi orthogonal-array engine
(:mod:`doexpert.doe.orthogonal_arrays`), which provides catalogue
lookup, automatic array selection, validation, and design-table
generation for mixed-level orthogonal arrays.
"""

from doexpert.doe.orthogonal_arrays import (  # noqa: F401
    OAInfo,
    auto_select_oa,
    build_design,
    can_accommodate,
    find_compatible_oas,
    generate_basic_oa_matrix,
    generate_design,
    get_oa_info,
    list_alternatives,
    list_catalog,
    search_oas_by_criteria,
)
