from typing import Self
from collections.abc import Callable, Sequence

from openff.qcsubmit.results.filters import (
    CMILESResultFilter,
    SinglepointRecordFilter,
    SinglepointRecordGroupFilter,
)
from openff.qcsubmit.results.results import (
    _BaseResult,
)
from openff.toolkit import Molecule
from qcportal.client import BaseRecord


class CustomResultFilter(CMILESResultFilter):
    """Custom filter that does not retrieve the record from QCArchive"""

    filter_function: Callable[[_BaseResult], bool]

    def _filter_function(self, result: "_BaseResult") -> bool:
        return self.filter_function(result)

    @classmethod
    def from_filter_function(cls, fn: Callable[[_BaseResult], bool]) -> Self:
        return cls(filter_function=fn)


class CustomSinglepointRecordFilter(SinglepointRecordFilter):
    """Custom filter that does retrieve the record from QCArchive"""

    filter_function: Callable[[_BaseResult, BaseRecord, Molecule], bool]

    def _filter_function(
        self,
        result: "_BaseResult",
        record: BaseRecord,
        molecule: Molecule,
    ) -> bool:
        return self.filter_function(result, record, molecule)

    @classmethod
    def from_filter_function(
        cls,
        fn: Callable[[_BaseResult, BaseRecord, Molecule], bool],
    ) -> Self:
        return cls(filter_function=fn)


class CustomSinglepointRecordGroupFilter(SinglepointRecordGroupFilter):
    """Custom filter which reduces repeated molecule entries down to a single
    entry.

    Notes:
        * This filter will only be applied to basic and optimization datasets.
          Torsion drive datasets / entries will be skipped.
    """

    filter_function: Callable[
        [Sequence[tuple["_BaseResult", BaseRecord, Molecule, str]]],
        list[tuple["_BaseResult", str]],
    ]

    def _filter_function(
        self,
        entries: Sequence[tuple["_BaseResult", BaseRecord, Molecule, str]],
    ) -> list[tuple["_BaseResult", str]]:
        return self.filter_function(entries)

    @classmethod
    def from_filter_function(
        cls,
        fn: Callable[
            [Sequence[tuple["_BaseResult", BaseRecord, Molecule, str]]],
            list[tuple["_BaseResult", str]],
        ],
    ) -> Self:
        return cls(filter_function=fn)
