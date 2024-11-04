import requests as rq
from datetime import datetime, timedelta
import pandas as pd

from typing import Dict
from dataclasses import dataclass
from pandas import DataFrame
from .base import BaseParser

@dataclass
class LentaParser(BaseParser):    
    def _get_url(self, param_dict: Dict[str, str]) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями
        """

        url = \
        "https://lenta.ru/search/v2/process?"\
        "from={offset}&"\
        "size={size}&"\
        "sort={sort}&"\
        "title_only={title_only}&"\
        "domain={domain}&"\
        "modified%2Cformat=yyyy-MM-dd&"\
        "modified%2Cfrom={date_from}&"\
        "modified%2Cto={date_to}&"\
        "query={query}"
        
        return url.format(**param_dict)


    def _get_search_table(self, param_dict: Dict[str, str]) -> DataFrame:
        """
        Возвращает DataFrame со списком статей
        """
        url = self._get_url(param_dict)
        r = rq.get(url)
        search_table = pd.DataFrame(r.json()["matches"])
        
        return search_table

    
    def get_request(
            self,
            param_dict: Dict[str, str],
            time_step: int = 10,
        ) -> DataFrame:
        """
        Функция для скачивания статей интервалами

        """
        params = param_dict.copy()
        time_step = timedelta(days = time_step)
        
        date_from = datetime.strptime(params["date_from"], "%Y-%m-%d")
        date_to = datetime.strptime(params["date_to"], "%Y-%m-%d")
        
        out = []
                
        while date_from <= date_to:
            if date_from + time_step > date_to:
                params["date_to"] = date_to.strftime("%Y-%m-%d")
            else:
                params["date_to"] = (date_from + time_step).strftime("%Y-%m-%d")
                
            print("Articles from {} to {}".format(params["date_from"], params["date_to"]))
            
            out.append(self._get_search_table(params))
            date_from += (time_step + timedelta(days = 1))
            params["date_from"] = date_from.strftime("%Y-%m-%d")

        print("Finish")
        
        return pd.concat(out, axis = 0, ignore_index = True)