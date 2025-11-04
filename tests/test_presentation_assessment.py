import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('/home/daytona/skill-training-llm/verbal_comm_skills_trainer')
from presentation_assessment import assess_presentation


class TestPresentationAssessment:
    """Test suite for presentation assessment functionality"""

    @pytest.fixture
    def mock_llm_wrapper(self):
        """Mock LLMWrapper for testing"""
        mock_llm = Mock()
        mock_llm.generate_response.return_value = (
            "Structure: Good introduction and conclusion, body could be more organized. "
            "Delivery: Minimal filler words, good pacing. "
            "Content: Persuasive arguments with strong vocabulary. "
            "Score: 8/10. Tips: Add more transitions between points."
        )
        return mock_llm

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_basic_functionality(self, mock_llm_class):
        """Test basic presentation assessment functionality"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Comprehensive presentation feedback"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "Good morning everyone. Today I will discuss the importance of teamwork."
        result = assess_presentation(input_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "Comprehensive presentation feedback"
        mock_llm_class.assert_called_once()
        mock_llm_instance.generate_response.assert_called_once()

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_text_preprocessing(self, mock_llm_class):
        """Test that input text is properly preprocessed (filler words removed)"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Processed feedback"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "Um, good morning everyone. Uh, today I will discuss teamwork."
        assess_presentation(input_text)
        
        # Verify the prompt was constructed with processed text
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        expected_processed_text = "good morning everyone. today i will discuss teamwork."
        
        assert "um" not in call_args.lower()
        assert "uh" not in call_args.lower()
        assert expected_processed_text in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_prompt_construction(self, mock_llm_class):
        """Test that the assessment prompt is constructed correctly"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Detailed assessment"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "This is a test presentation."
        assess_presentation(input_text)
        
        # Verify the prompt contains all required assessment criteria
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        
        assert "structure (introduction, body, conclusion)" in call_args.lower()
        assert "delivery (pacing, filler words)" in call_args.lower()
        assert "content (persuasiveness, vocabulary)" in call_args.lower()
        assert "scores and actionable tips" in call_args.lower()
        assert "this is a test presentation." in call_args.lower()

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_with_empty_input(self, mock_llm_class):
        """Test assessment with empty input text"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Please provide presentation content to assess."
        mock_llm_class.return_value = mock_llm_instance
        
        result = assess_presentation("")
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify prompt was still constructed properly
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        assert "presentation text: " in call_args.lower()

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_with_whitespace_only(self, mock_llm_class):
        """Test assessment with whitespace-only input"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "No meaningful content to assess."
        mock_llm_class.return_value = mock_llm_instance
        
        result = assess_presentation("   \n\t   ")
        
        assert isinstance(result, str)
        # Verify whitespace is processed correctly
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        assert "presentation text:    \n\t   " in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_with_multiple_filler_words(self, mock_llm_class):
        """Test preprocessing with multiple types of filler words"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Clean presentation feedback"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "Um, so, uh, today we will, um, discuss the, uh, importance of communication."
        assess_presentation(input_text)
        
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        processed_text = "so, today we will, discuss the, importance of communication."
        
        assert processed_text in call_args
        assert "um" not in call_args.lower()
        assert "uh" not in call_args.lower()

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_case_insensitive_processing(self, mock_llm_class):
        """Test that text processing handles different cases correctly"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Case-processed feedback"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "UM, Good Morning Everyone. UH, Today's Topic Is Important."
        assess_presentation(input_text)
        
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        expected_processed = "good morning everyone. today's topic is important."
        
        assert expected_processed in call_args
        assert "UM" not in call_args
        assert "UH" not in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_with_long_text(self, mock_llm_class):
        """Test assessment with longer presentation text"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Comprehensive analysis of long presentation"
        mock_llm_class.return_value = mock_llm_instance
        
        long_text = """
        Good morning everyone. Today I want to talk about the importance of effective communication.
        First, let me introduce the main points. Communication is essential for teamwork.
        In the body of my presentation, I will cover three key areas: verbal communication,
        non-verbal communication, and written communication. Each of these plays a crucial role.
        In conclusion, effective communication is the foundation of successful collaboration.
        Thank you for your attention.
        """
        
        result = assess_presentation(long_text)
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Verify the full text was processed
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        assert "good morning everyone" in call_args
        assert "thank you for your attention" in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_llm_error_handling(self, mock_llm_class):
        """Test handling of LLM generation errors"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.side_effect = Exception("LLM generation failed")
        mock_llm_class.return_value = mock_llm_instance
        
        with pytest.raises(Exception, match="LLM generation failed"):
            assess_presentation("Test presentation content")

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_llm_initialization_error(self, mock_llm_class):
        """Test handling of LLMWrapper initialization errors"""
        mock_llm_class.side_effect = Exception("Failed to initialize LLM")
        
        with pytest.raises(Exception, match="Failed to initialize LLM"):
            assess_presentation("Test presentation content")

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_with_special_characters(self, mock_llm_class):
        """Test assessment with special characters and punctuation"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Special character handling feedback"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "Hello! Today's topic is AI & ML... It's fascinating, isn't it? (Rhetorical question)"
        assess_presentation(input_text)
        
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        expected_processed = "hello! today's topic is ai & ml... it's fascinating, isn't it? (rhetorical question)"
        
        assert expected_processed in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_with_numbers_and_dates(self, mock_llm_class):
        """Test assessment with numerical content and dates"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Numerical content assessment"
        mock_llm_class.return_value = mock_llm_instance
        
        input_text = "In 2023, we achieved 95% customer satisfaction. Our revenue increased by $2.5 million."
        assess_presentation(input_text)
        
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        expected_processed = "in 2023, we achieved 95% customer satisfaction. our revenue increased by $2.5 million."
        
        assert expected_processed in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_multiple_calls_independence(self, mock_llm_class):
        """Test that multiple assessment calls are independent"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.side_effect = [
            "First assessment result",
            "Second assessment result"
        ]
        mock_llm_class.return_value = mock_llm_instance
        
        result1 = assess_presentation("First presentation content")
        result2 = assess_presentation("Second presentation content")
        
        assert result1 == "First assessment result"
        assert result2 == "Second assessment result"
        assert mock_llm_instance.generate_response.call_count == 2
        
        # Verify different content was processed
        call_args_list = mock_llm_instance.generate_response.call_args_list
        assert "first presentation content" in call_args_list[0][0][0]
        assert "second presentation content" in call_args_list[1][0][0]

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_filler_word_edge_cases(self, mock_llm_class):
        """Test edge cases in filler word removal"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Edge case handling feedback"
        mock_llm_class.return_value = mock_llm_instance
        
        # Test words containing "um" and "uh" but not as standalone filler words
        input_text = "The museum has a unique atmosphere. The hummingbird is beautiful."
        assess_presentation(input_text)
        
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        # These words should NOT be affected by filler word removal
        assert "museum" in call_args
        assert "unique" in call_args
        assert "atmosphere" in call_args
        assert "hummingbird" in call_args

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_return_type_validation(self, mock_llm_class):
        """Test that the function always returns a string"""
        mock_llm_instance = Mock()
        
        # Test with different return types from LLM
        test_cases = [
            "Normal string response",
            "",  # Empty string
            "   ",  # Whitespace string
        ]
        
        for expected_response in test_cases:
            mock_llm_instance.generate_response.return_value = expected_response
            mock_llm_class.return_value = mock_llm_instance
            
            result = assess_presentation("Test input")
            assert isinstance(result, str)
            assert result == expected_response

    def test_assess_presentation_function_signature(self):
        """Test that the function has the expected signature"""
        import inspect
        
        sig = inspect.signature(assess_presentation)
        params = list(sig.parameters.keys())
        
        assert len(params) == 1
        assert params[0] == "input_text"
        
        # Verify function is callable
        assert callable(assess_presentation)

    @patch('presentation_assessment.LLMWrapper')
    def test_assess_presentation_prompt_completeness(self, mock_llm_class):
        """Test that the assessment prompt includes all required evaluation criteria"""
        mock_llm_instance = Mock()
        mock_llm_instance.generate_response.return_value = "Complete assessment"
        mock_llm_class.return_value = mock_llm_instance
        
        assess_presentation("Sample presentation text")
        
        call_args = mock_llm_instance.generate_response.call_args[0][0]
        
        # Verify all assessment dimensions are included
        required_criteria = [
            "structure",
            "introduction",
            "body", 
            "conclusion",
            "delivery",
            "pacing",
            "filler words",
            "content",
            "persuasiveness",
            "vocabulary",
            "scores",
            "actionable tips"
        ]
        
        for criterion in required_criteria:
            assert criterion.lower() in call_args.lower(), f"Missing criterion: {criterion}"
