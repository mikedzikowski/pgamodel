from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class Player(BaseModel):
    dg_id: int
    player_name: str
    country: str = ""
    amateur: int = 0


class Tournament(BaseModel):
    event_id: int
    event_name: str
    course: str = ""
    date: str = ""
    tour: str = "pga"
    purse: Optional[float] = None


class Prediction(BaseModel):
    player_name: str
    dg_id: int
    win_prob: float = Field(ge=0.0, le=1.0)
    top5_prob: float = Field(ge=0.0, le=1.0)
    top10_prob: float = Field(ge=0.0, le=1.0)
    top20_prob: float = Field(ge=0.0, le=1.0)
    make_cut_prob: float = Field(ge=0.0, le=1.0)

    @property
    def expected_value(self) -> float:
        return (
            self.win_prob * 1.0
            + self.top5_prob * 0.4
            + self.top10_prob * 0.2
            + self.top20_prob * 0.1
        )


class OaDPick(BaseModel):
    player_name: str
    dg_id: int
    event_name: str
    event_id: int
    win_prob: float
    expected_value: float
