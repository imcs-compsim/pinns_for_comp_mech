from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import pandas as pd


class ExperimentLogger:

    def __init__(self, save_path: str, append: bool = True, autosave: bool=True):
        # path where the logs should be stored
        self._path = Path(save_path).resolve().expanduser()
        # whether to append to existing logs or start fresh
        self._append = append
        # whether to automatically save logs after each recording
        self._autosave = autosave
        # initialize an empty DataFrame to store experiment logs
        self._data = pd.DataFrame()
        self._current_row = {}
        self._start = -1.
        self._last = -1.

    def __enter__(self):
        """This function is automatically invoked upon entering the context.
        
        """
        self._start = timer()
        self._last = self._start
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """This function is automatically invoked upon leaving the context.

        TODO

        Parameters
        ----------
        exc_type : Any
            The type of an exception that occured in the context; defaults to 
            ``None`` if no exception was thrown.
        exc_val : Any
            The exception instance that was thrown in the context; defaults to 
            ``None`` if no exception was thrown.
        exc_tb : Any
            The traceback of the exception that was thrown in the context; 
            defaults to ``None`` if no exception was thrown.

        Returns
        -------
        bool
            Flag indicating whether or not to surpress exceptions. Always 
            ``False`` for this class, meaning that exceptions will never be
            surpressed.
        """
        self.log_time('total', rel_to_prev=False)
        if self._autosave:
            self.save()
        return False
    
    def _add_row(self):
        if self._current_row:
            self._data = pd.concat(
                [self._data, pd.DataFrame([self._current_row])],
                ignore_index=True
            )
            self._current_row = {}

    def new_entry(self, keyval: float=None, keyname: str="key"):
        self._add_row()
        if keyval is not None:
            self._current_row[keyname] = keyval

    def log_time(self, what: str, rel_to_prev: bool=True):
        now = timer()
        self._current_row[f'{what}_time'] = now - self._last if rel_to_prev else now - self._start
        self._last = now

    def log_metric(self, name: str, value: float):
        self._current_row[name] = value

    def save(self): 
        self._add_row()
        self._data.to_hdf(self._path, key='experiment_logs', mode='a' if self._append else 'w')
