# Executa ao importar qualquer parte do projeto (servidor, ingestão ou avaliação):
# deixa o Python confiar nos certificados do sistema quando a rede intercepta o HTTPS.
from app import tls as _tls

_tls.injetar()
