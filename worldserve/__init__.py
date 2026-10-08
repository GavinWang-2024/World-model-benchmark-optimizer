"""`worldserve`: the serving layer, under the name the outline uses.

    from worldserve import WorldModelServer

The implementation lives in `worldoptbench.serve` (this is only an alias, so there is one copy of the code);
the command line is `worldserve` (see pyproject.toml) or `python -m worldoptbench.serve`.
"""

from worldoptbench.serve import (
    ServerBusy,
    ServerStopped,
    WorldModelServer,
    build_model,
    main,
    make_http_server,
)

__all__ = ["ServerBusy", "ServerStopped", "WorldModelServer", "build_model", "main", "make_http_server"]
