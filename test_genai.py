import unittest
from genai_qa import ABSTAIN, answer, retrieve, prompt_for

class GenAITests(unittest.TestCase):
    def test_no_overlap_does_not_generate(self):
        def forbidden(_):
            raise AssertionError('Generation must not run')
        result = answer('zzzxxyy', forbidden)
        self.assertEqual(result['answer'], ABSTAIN)
        self.assertEqual(result['sources'], [])

    def test_invalid_questions(self):
        for value in ['', ' ', None, 2, 'x'*501]:
            with self.assertRaises(ValueError):
                retrieve(value)

    def test_generation_receives_actual_context(self):
        prompts = []
        result = answer('Sobel edges', lambda p: prompts.append(p) or 'model output')
        self.assertEqual(result['answer'], 'model output')
        self.assertIn('Sobel', prompts[0])
        self.assertEqual(result['status'], 'generated_unverified')
        self.assertLessEqual(len(result['sources']), 2)

    def test_empty_corpus(self):
        self.assertEqual(retrieve('test', []), [])

    def test_prompt_has_question(self):
        self.assertIn('Question: why?', prompt_for('why?', [{'text':'Evidence'}]))

if __name__ == '__main__':
    unittest.main()
