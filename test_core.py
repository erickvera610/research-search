import unittest
from core import search
class Tests(unittest.TestCase):
    def test_relevant_ranking(self): self.assertEqual(search('logistic classification')[0]['title'],'Machine learning validation')
    def test_unknown_term(self): self.assertEqual(search('xyzabc'),[])
    def test_case_insensitive(self): self.assertEqual(search('SQLITE'),search('sqlite'))
    def test_empty(self):
        with self.assertRaises(ValueError): search(' ')
    def test_exact_match(self): self.assertEqual(search('hello',[{'title':'hello','text':'hello'}])[0]['score'],1)

    def test_title_retrieval(self): self.assertEqual(search('machine learning validation')[0]['title'],'Machine learning validation')
