def check_T(T: int):
    if T < 2:
        raise ValueError(
            f'T must be an integer larger than 1, but got `{T}`'
        )
