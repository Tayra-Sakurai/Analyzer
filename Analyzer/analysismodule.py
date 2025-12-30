"""The module for analysis."""
import numpy as np
from lammpsio import DumpFile, Snapshot
from typing import Any, Iterable, Literal, TypeVar, NotRequired, TypedDict
import numpy.typing as npt
from scipy.optimize import lsq_linear
from os import PathLike

S = TypeVar("S")

type FileType = PathLike | str


SchemaType = TypedDict('SchemaType',{
    "id": NotRequired[int],
    "typeid": NotRequired[int],
    "molecule": NotRequired[int],
    "charge": NotRequired[int],
    "mass": NotRequired[int],
    "position": NotRequired[tuple[int, int, int]],
    "velocity": NotRequired[tuple[int, int, int]],
    "image": NotRequired[tuple[int, int, int]]
})


def end_vector(
    snapshot: Snapshot
) -> npt.NDArray[np.floating[Any]]:
    """Returns the end to end vector.

    Parameters
    ----------
    snapshot : Snapshot
        The `Snapshot` instance.

    Returns
    -------
    array : NDArray[floating[Any]]
        The vector.
    """
    pos: npt.NDArray[np.floating[Any]] = snapshot.position
    return pos[-1] - pos[0]


def to_cvec(
    vec: np.ndarray[
        tuple[int, ...],
        S
    ]
) -> np.ndarray[
    tuple[int, int],
    S
]:
    """Converts to a column vector.

    Parameters
    ----------
    vec : NDArray[floating[Any]]

    Returns
    -------
    result : NDArray[floating[Any]]
        The column vector.
    """
    return vec.reshape((-1, 1))


def regress(
    nvec: npt.NDArray[np.floating[Any] | np.integer[Any]],
    yvec: npt.NDArray[np.floating[Any]]
) -> tuple[
    np.floating[Any] | float,
    np.floating[Any] | float
]:
    """Regresses a line to the logarithm values of the distances.

    Parameters
    ----------
    nvec : NDArray[floating[Any] | integer[Any]]
        The values of `n`s.
    yvec : NDArray[floating[Any]]
        The values of `Re`s.

    Returns
    -------
    result : tuple[floating | float, ...]
        Two-value tuple which contains the `a` value
        which fulfills `y = a log10(n - 1)` and the error of it.
    """
    vecs = nvec - 1, yvec
    vecs = list(map(np.log10, vecs))
    columnvecs = to_cvec(vecs[0]), vecs[1]
    res = lsq_linear(*columnvecs)
    dy = np.sum((vecs[1] - res.x[0] * vecs[0]) ** 2)
    dy /= len(yvec) - 2
    dx = 1 / np.sum(vecs[0] ** 2)
    delta = dy * dx
    delta = np.sqrt(delta)
    return res.x[0], delta


class ProcessAnalysis:
    """Main class for any analysis.

    Parameters
    ----------
    files : Iterable[FileType]
        Files to analyze.
    schema : SchemaType
    """
    def __init__(
        self,
        files: Iterable[FileType],
        schema: SchemaType
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        files : Iterable[FileType]
            The files.
        schema : SchemaType
            The content schema.
        """
        flist = map(str, files)
        self.dumps: list[DumpFile] = list()
        self.schema = schema
        for f in flist:
            self.dumps.append(DumpFile(f, schema))

    def get_points(
        self,
        count: int
    ) -> npt.NDArray[np.floating[Any]]:
        """Gets the point data.

        Parameters
        ----------
        count : int
            The number of timesteps.

        Returns
        -------
        result : NDArray[flosting[Any]]
        """
        nl: list[int] = list()
        rl: list[float | np.floating[Any]] = list()
        for dump in self.dumps:
            snaplist: list[Snapshot] = sorted(
                dump,
                key=lambda snap: snap.step
            )
            nl.append(snaplist[0].N)
            rvec = map(end_vector, snaplist[-count:])
            rmat = np.array(list(rvec))
            norms = np.sum(rmat ** 2, 1)
            rl.append(np.mean(norms))
        return np.array((nl, rl))

    def regress(
        self,
        count: int = 5
    ) -> tuple[
        np.floating[Any] | float,
        np.floating[Any] | float
    ]:
        """Gets the line.

        Parameters
        ----------
        count : int, optional

        Returns
        -------
        result : tuple[floating[Any] | float, floating[Any] | float]

        See Also
        --------
        analysismodule.regress : for detail
        """
        ps = self.get_points(count)
        return regress(ps[0], ps[1])
