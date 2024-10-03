import pathlib
from dataclasses import dataclass
from typing import Any, Dict

from langchain_community.chat_models.gigachat import GigaChat
from langchain_community.embeddings.gigachat import GigaChatEmbeddings


GIGA_CRED = "MGY5ZDg4YzAtYzRhNS00ZGYyLTk5Y2ItZjQ0ZTFjYTBjZGY5OjJhZDllNGI4LTAwOTQtNDQzNi05ZWQwLWQwN2VjMzU5MWFmYg=="
# GIGA_CRED = "MGY5ZDg4YzAtYzRhNS00ZGYyLTk5Y2ItZjQ0ZTFjYTBjZGY5OjIxYjFhMzY1LTJiOWEtNGEwNC1hYjhlLThkNDRhMzY4NGFjYw=="
# TG_TOKEN = "7576366730:AAFshsh7L6xV--UAsB5Xy3mhsjkHbsb5p_k"
TG_TOKEN = "7269551208:AAG1biO273Vf8Qc4Z1AMl3k-8kuWlgThNq4"  # naer


VERBOSE: bool = False

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 70

DOC_PATH = str(pathlib.Path(__file__).parent.parent / "data")

TEST_CP_GOOD_PATH = str(pathlib.Path(__file__).parent / "mocks" / "cp_good.json")
TEST_CP_NOT_CACL_PATH = str(pathlib.Path(__file__).parent / "mocks" / "cp_not_calc.json")
TEST_CP_BAD_PATH = str(pathlib.Path(__file__).parent / "mocks" / "cp_bad.json")


@dataclass
class GigaSettings:
    stand: str = "ext"  # "ext" - внешний(для тестов), "ift", "uat", "prod"

    def __post_init__(self):
        model_options: Dict[str, Any] = {
            "model": "GigaChat-Pro",
            "top_p": 0.1,
            "repetition_penalty": 0.9,
            "timeout": 50.0,
            "verbose": VERBOSE,
            "temperature": 1e-8,
            # "max_tokens": 125,
        }

        self.stand = self.stand.lower()
        if self.stand == "ext":
            base_url = "https://gigachat.devices.sberbank.ru/api/v1"
            auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
            giga_cred = GIGA_CRED
            scope = "GIGACHAT_API_CORP"

            model_options = {
                **model_options,
                "base_url": base_url,
                "auth_url": auth_url,
                "credentials": giga_cred,
                "scope": scope,
                "verify_ssl_certs": False,
            }

            embdeddings_options = {
                "base_url": base_url,
                "auth_url": auth_url,
                "credentials": giga_cred,
                "scope": scope,
                "verify_ssl_certs": False,
            }

        else:
            if self.stand == "ift":
                base_url = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1"
            elif self.stand == "psi" or self.stand == "uat":
                base_url = "https://gigachat-psi.sberdevices.ca.sbrf.ru/v1"
            else:  # prod
                base_url = "https://gigachat-prom.sberdevices.ca.sbrf.ru/v1"
                self.stand = "prod"
            ca_cert_path = str(pathlib.Path(__file__).parent / "cert" / self.stand / "ca.pem")
            cert_path = str(pathlib.Path(__file__).parent / "cert" / self.stand / "cert.pem")
            cert_key_path = str(pathlib.Path(__file__).parent / "cert" / self.stand / "cert.key")

            model_options = {
                **model_options,
                "base_url": base_url,
                "ca_bundle_file": ca_cert_path,  # chain_pem.txt
                "cert_file": cert_path,  # published_pem.txt
                "key_file": cert_key_path,
            }

            embdeddings_options = {
                "base_url": base_url,
                "ca_bundle_file": ca_cert_path,  # chain_pem.txt
                "cert_file": cert_path,  # published_pem.txt
                "key_file": cert_key_path,
            }

        self.chat_model = GigaChat(**model_options)
        self.embeddings = GigaChatEmbeddings(**embdeddings_options)
