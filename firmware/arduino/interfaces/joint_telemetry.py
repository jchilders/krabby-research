from dataclasses import dataclass
from typing import Tuple, Optional

# Telemetry schema (matches interfaces/joint_telemetry.h):
# Single line contains all joints, segments separated by ';'
# Line example:
#   JT LHY 0.123 0 12 1 1 0 120 0;RHY ...;LHL ...
# Segment format:
#   <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf>
# Names: LHY, RHY, LHL, LKL, RHL, RKL


@dataclass
class JointTelemetry:
    name: str
    pos: float
    pot: int
    current: int
    en: Tuple[int, int]
    pwm: Tuple[int, int]
    saf: int

    @classmethod
    def from_tokens(cls, tokens) -> Optional["JointTelemetry"]:
        if not tokens:
            return None
        if tokens[0] == "JT":
            tokens = tokens[1:]
        if len(tokens) != 9:
            return None
        name, pos, pot, cur, enL, enR, pwmL, pwmR, saf = tokens
        try:
            return cls(
                name=name,
                pos=float(pos),
                pot=int(pot),
                current=int(cur),
                en=(int(enL), int(enR)),
                pwm=(int(pwmL), int(pwmR)),
                saf=int(saf),
            )
        except ValueError:
            return None

    @classmethod
    def parse_line(cls, line: str):
        joints = []
        for seg in line.strip().split(";"):
            seg = seg.strip()
            if not seg:
                continue
            tokens = seg.split()
            jt = cls.from_tokens(tokens)
            if jt:
                joints.append(jt)
        return joints

    def format_compact(self, target: Optional[float] = None) -> str:
        pos_part = f"{self.pos:.3f}"
        if target is not None:
            pos_part = f"{pos_part}/{target:.3f}"
        return (
            f"{self.name}:{pos_part},{self.pot},{self.current},"
            f"({self.en[0]},{self.en[1]}),({self.pwm[0]},{self.pwm[1]}),{self.saf}"
        )
