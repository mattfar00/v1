"""backtest.data — estrazione delle serie storiche reali per il backtest."""
from .series import (
    serie_fondo_reale,
    serie_pac_reale,
    serie_singolo_asset,
    allinea_ultimi_mesi,
    allinea_coppia,
)

__all__ = [
    "serie_fondo_reale", "serie_pac_reale", "serie_singolo_asset",
    "allinea_ultimi_mesi", "allinea_coppia",
]
