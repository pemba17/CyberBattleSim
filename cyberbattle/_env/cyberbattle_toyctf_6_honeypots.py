# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..samples.toyctf import toy_ctf_6_honeypots
from . import cyberbattle_env


class CyberBattleToyCtf6HoneyPots(cyberbattle_env.CyberBattleEnv):
    """CyberBattle simulation based on a toy CTF exercise"""

    def __init__(self, **kwargs):
        super().__init__(initial_environment=toy_ctf_6_honeypots.new_environment(), **kwargs)
