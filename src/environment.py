from dataclasses import dataclass


@dataclass
class Environment:
    stand: str
    os: str
    sbol_version: str
