"""Certificados HTTPS do sistema operacional (resolve CERTIFICATE_VERIFY_FAILED em redes com proxy).

O problema: redes de faculdade/empresa costumam "abrir" o HTTPS com um certificado próprio.
Windows e Chrome confiam nele; o Python, por padrão, não (usa o pacote `certifi`). Resultado:
`SSL: CERTIFICATE_VERIFY_FAILED ... unable to get local issuer certificate`.

A solução: usar o repositório de certificados do próprio sistema (biblioteca `truststore`).
- Automático no Windows e no macOS. No Linux (servidor de deploy) nada muda.
- USE_SYSTEM_CERTS=1 força ligar; USE_SYSTEM_CERTS=0 força desligar.
"""

from __future__ import annotations

import logging
import os
import ssl
import sys

from dotenv import load_dotenv

log = logging.getLogger(__name__)


def ativo() -> bool:
    load_dotenv()  # permite definir USE_SYSTEM_CERTS no arquivo .env
    valor = os.getenv("USE_SYSTEM_CERTS", "auto").strip().lower()
    if valor in ("1", "true", "sim", "yes"):
        return True
    if valor in ("0", "false", "nao", "não", "no"):
        return False
    return sys.platform in ("win32", "darwin")


def injetar() -> bool:
    """Faz as bibliotecas que usam `ssl` (requests, google-auth...) confiarem no repositório do sistema."""
    if not ativo():
        return False
    try:
        import truststore
    except ImportError:
        log.warning("USE_SYSTEM_CERTS ativo, mas o pacote 'truststore' não está instalado (pip install truststore).")
        return False
    truststore.inject_into_ssl()
    return True


def contexto() -> ssl.SSLContext | None:
    """Contexto SSL que usa os certificados do sistema, para passar ao cliente HTTP do Gemini."""
    if not ativo():
        return None
    try:
        import truststore
    except ImportError:
        return None
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
