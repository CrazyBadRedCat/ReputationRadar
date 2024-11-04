import requests as rq
from datetime import datetime, timedelta
import pandas as pd
from bs4 import BeautifulSoup as bs

from typing import Dict
from dataclasses import dataclass
from pandas import DataFrame
from .base import BaseParser

@dataclass
class RbcParser(BaseParser):
    max_page: int = 10

    def _get_url(self, param_dict: Dict[str, str]) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями
        """

        url = \
        "https://www.rbc.ru/search/ajax/?"\
        "project={project}&"\
        "category={category}&"\
        "dateFrom={date_from}&"\
        "dateTo={date_to}&"\
        "page={page}&"\
        "query={query}&"\
        "material={material}"
        
        return url.format(**param_dict)


    def _get_search_table(
        self,
        param_dict: Dict[str, str],
        include_text: bool = True,
    ) -> DataFrame:
        """
        Возвращает DataFrame со списком статей
        """
        url = self._get_url(param_dict)
        r = rq.get(url)
        search_table = pd.DataFrame(r.json()["items"])
        
        if include_text and not search_table.empty:
            get_text = lambda x: self._get_article_data(x["fronturl"])
            search_table[["overview", "text"]] = \
            search_table.apply(get_text, axis = 1).tolist()
        
        if "publish_date_t" in search_table.columns:
            search_table.sort_values("publish_date_t", ignore_index = True)
            
        return search_table

    def _get_article_data(self, url: str):
        """
        Возвращает описание и текст статьи по ссылке
        """
        r = rq.get(url)
        soup = bs(r.text, features = "lxml")
        div_overview = soup.find("div", {"class": "article__text__overview"})

        if div_overview:
            overview = div_overview.text.replace("<br />", "\n").strip()
        else:
            overview = None

        p_text = soup.find_all("p")
        if p_text:
            text = ' '.join(
                map(lambda x: x.text.replace("<br />", "\n").strip(), p_text)
            )
        else:
            text = None
        
        return overview, text
    
    def _iterable_load_by_page(
        self,
        param_dict: Dict[str, str],
        include_text: bool = True,
    ) -> DataFrame:
        params = param_dict.copy()
        results = []
        
        result = self._get_search_table(params)
        results.append(result)
        
        while not result.empty:
            if int(params["page"]) >= self.max_page:
                break
            
            params["page"] = str(int(params["page"]) + 1)
            result = self._get_search_table(params, include_text)
            results.append(result)
                    
        return pd.concat(results, axis = 0, ignore_index = True)
    
    
    def get_request(
            self,
            param_dict: Dict[str, str],
            time_step: int = 10,
            include_text: bool = True,
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
            
            out.append(self._iterable_load_by_page(params, include_text))
            date_from += (time_step + timedelta(days = 1))
            params["date_from"] = date_from.strftime("%Y-%m-%d")

        print("Finish")
        
        return pd.concat(out, axis = 0, ignore_index = True)