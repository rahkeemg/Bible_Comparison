import requests
import json

class My_API:
    """
        This is a custom class created to access API sites using Python OOP.
    """
    
    def __init__(self, url=None, key='', query=''):
        self.url = url
        self._key = key
        if key:
            self.set_key(key)        
        self.query = query
        self._query_string = ''
        
    def set_key(self, key=None):
        if key:
            self._key = f"&key={key}"
        
    def set_url(self, url=None):
        if url:
            self.url = url            
        
    def run_query(self, query, verbose=False):
        self._query_string = My_API.build_query(self.url, query, self._key)
        self.resp = requests.get(self._query_string)
        try:
            data = self.resp.json()
        except:
            if verbose:
                print("---------------------------------------------------------------")
                print("Error converting response to JSON format. Returning as a string")
                print("---------------------------------------------------------------")
            data = self.resp.text
        return data

    def clean_convert_response(self):
        pass
        
    
    @staticmethod
    def build_query(url='', query='', key=''):
        q = ''.join([url,query,key])
        return q