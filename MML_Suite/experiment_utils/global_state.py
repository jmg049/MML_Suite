CURRENT_RUN_ID: int = None
CURRENT_EXP_NAME: str = None
DEBUG: bool = True


def set_current_run_id(run_id: int):
    global CURRENT_RUN_ID

    CURRENT_RUN_ID = run_id
    print(f"Current run ID set to: {CURRENT_RUN_ID}")


def set_current_exp_name(exp_name: str):
    global CURRENT_EXP_NAME
    CURRENT_EXP_NAME = exp_name


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


def get_current_run_id() -> int:
    return CURRENT_RUN_ID


def get_current_exp_name() -> str:
    return CURRENT_EXP_NAME


def get_debug() -> bool:
    return DEBUG
