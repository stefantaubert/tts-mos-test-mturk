from typing import OrderedDict as ODType
from typing import Union

WorkerName = str
MaskName = str
AlgorithmName = str
AssignmentId = str
FileName = str
RatingName = str
RatingValue = Union[int, float]
Ratings = ODType[RatingName, RatingValue]
