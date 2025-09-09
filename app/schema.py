from pydantic import BaseModel
from typing import Any, List, Optional, Dict


class FieldValue(BaseModel):
    value: Optional[Any]
    confidence: float
    bbox: Optional[List[int]] = None


class SubjectRow(BaseModel):
    subject_name: FieldValue
    max_marks: Optional[FieldValue]
    obtained_marks: Optional[FieldValue]
    grade: Optional[FieldValue]


class Candidate(BaseModel):
    name: Optional[FieldValue]
    father_name: Optional[FieldValue]
    mother_name: Optional[FieldValue]
    dob: Optional[FieldValue]
    roll_no: Optional[FieldValue]
    registration_no: Optional[FieldValue]
    exam_year: Optional[FieldValue]
    board: Optional[FieldValue]
    institution: Optional[FieldValue]


class Overall(BaseModel):
    total_max_marks: Optional[FieldValue]
    total_obtained: Optional[FieldValue]
    percentage: Optional[FieldValue]
    result: Optional[FieldValue]
    grade: Optional[FieldValue]


class Issue(BaseModel):
    issue_date: Optional[FieldValue]
    issue_place: Optional[FieldValue]


class ExtractionResponse(BaseModel):
    document_id: str
    source: Dict[str, Any]
    candidate: Candidate
    subjects: List[SubjectRow]
    overall: Optional[Overall]
    issue: Optional[Issue]
    extraction_meta: Dict[str, Any]
    errors: List[str] = []