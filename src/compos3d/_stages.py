# Shared helpers for staged command placeholders.

class StagePendingError(RuntimeError):
    pass


def stage_pending(capability: str, stage: int) -> None:
    raise StagePendingError(
        f'{capability} is scheduled for Stage {stage}. '
        'Stage 1 only exposes the cleaned research-facing command surface.'
    )
