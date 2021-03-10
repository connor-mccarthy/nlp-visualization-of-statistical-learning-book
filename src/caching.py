import os
import pickle
from datetime import datetime
from typing import Any, Callable


def cache(cacheable_func: Callable) -> Callable:
    def json_getter_wrapper(*args: Any, **kwargs: Any) -> Any:
        cache_path = get_cache_path(*args, **kwargs)
        use_cache = should_use_cache(cache_path)

        if use_cache:
            print("Using cache.")
            with open(cache_path, "rb") as pickle_handle:
                result = pickle.load(pickle_handle)
        else:
            result = cacheable_func(*args, **kwargs)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb+") as pickle_handle:
                pickle.dump(result, pickle_handle)

        return result

    return json_getter_wrapper


def get_cache_path(*args: Any, **kwargs: Any) -> str:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    cache_folder = ".cache"
    string_args = str(args) if args else ""
    string_kwargs = str(kwargs) if kwargs else ""
    filename = string_args + string_kwargs
    replacements = [("/", ":"), (" ", ""), ("{", ""), ("}", ""), ("'", "")]
    for replacement in replacements:
        filename = filename.replace(*replacement)
    return os.path.join(current_directory, cache_folder, filename + ".pickle")


def should_use_cache(cache_path: str) -> bool:
    file_exists = os.path.exists(cache_path)
    use_cache = False

    if file_exists:
        created_time = datetime.fromtimestamp(os.path.getctime(cache_path))
        age_seconds = (datetime.now() - created_time).seconds
        second_per_hour = 60 * 60
        age_hours = age_seconds / second_per_hour

        CACHE_LIFESPAN_HOURS = 0.1
        if age_hours < CACHE_LIFESPAN_HOURS:
            use_cache = True

    return use_cache


@cache
def example_cacheable_func(val: Any) -> Any:
    return val


if __name__ == "__main__":
    print(example_cacheable_func(val="some_val_goes_here!"))
