import pytest
import gradio as gr
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('/home/daytona/skill-training-llm/verbal_comm_skills_trainer')
import chat_interface
from chat_interface import chat_response


class TestChatInterface:
    """Test suite for chat interface functionality"""

    @pytest.fixture
    def mock_llm_wrapper(self):
        """Mock LLMWrapper for testing"""
        mock_llm = Mock()
        mock_llm.generate_response.return_value = "Expert communication feedback provided"
        return mock_llm

    @patch('chat_interface.llm')
    def test_chat_response_basic_functionality(self, mock_llm):
        """Test basic chat response functionality"""
        mock_llm.generate_response.return_value = "Great communication! Your message is clear and well-structured."
        
        user_input = "I feel nervous about public speaking."
        result = chat_response(user_input)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert result == "Great communication! Your message is clear and well-structured."
        
        # Verify the correct prompt was constructed
        expected_prompt = (
            "You are a communication expert. Provide detailed feedback on clarity, tone, "
            "and suggestions for improvement. Here is the user's input: " + user_input
        )
        mock_llm.generate_response.assert_called_once_with(expected_prompt)

    @patch('chat_interface.llm')
    def test_chat_response_prompt_construction(self, mock_llm):
        """Test that chat prompts are properly constructed"""
        mock_llm.generate_response.return_value = "Feedback response"
        
        user_input = "How can I improve my presentation skills?"
        chat_response(user_input)
        
        call_args = mock_llm.generate_response.call_args[0][0]
        
        # Verify all required prompt components are present
        assert "You are a communication expert" in call_args
        assert "Provide detailed feedback on clarity, tone" in call_args
        assert "suggestions for improvement" in call_args
        assert "Here is the user's input:" in call_args
        assert user_input in call_args

    @patch('chat_interface.llm')
    def test_chat_response_with_empty_input(self, mock_llm):
        """Test chat response with empty user input"""
        mock_llm.generate_response.return_value = "Please provide some input to analyze."
        
        result = chat_response("")
        
        assert isinstance(result, str)
        assert result == "Please provide some input to analyze."
        
        # Verify prompt was constructed with empty input
        call_args = mock_llm.generate_response.call_args[0][0]
        assert call_args.endswith("Here is the user's input: ")

    @patch('chat_interface.llm')
    def test_chat_response_with_whitespace_input(self, mock_llm):
        """Test chat response with whitespace-only input"""
        mock_llm.generate_response.return_value = "I notice your input contains only whitespace."
        
        user_input = "   \n\t   "
        result = chat_response(user_input)
        
        assert isinstance(result, str)
        assert result == "I notice your input contains only whitespace."
        
        # Verify whitespace was preserved in prompt
        call_args = mock_llm.generate_response.call_args[0][0]
        assert user_input in call_args

    @patch('chat_interface.llm')
    def test_chat_response_with_long_input(self, mock_llm):
        """Test chat response with very long user input"""
        mock_llm.generate_response.return_value = "Comprehensive feedback on your detailed message."
        
        # Create a long input (over 1000 characters)
        long_input = "I have been struggling with public speaking for years. " * 25
        result = chat_response(long_input)
        
        assert isinstance(result, str)
        assert result == "Comprehensive feedback on your detailed message."
        
        # Verify the full long input was included in prompt
        call_args = mock_llm.generate_response.call_args[0][0]
        assert long_input in call_args
        assert len(call_args) > 1000

    @patch('chat_interface.llm')
    def test_chat_response_with_special_characters(self, mock_llm):
        """Test chat response with special characters and punctuation"""
        mock_llm.generate_response.return_value = "Good use of punctuation and emphasis!"
        
        user_input = "Hello! How are you? I'm excited about this: 1) learning, 2) growing & 3) improving!"
        result = chat_response(user_input)
        
        assert isinstance(result, str)
        assert result == "Good use of punctuation and emphasis!"
        
        # Verify special characters were preserved
        call_args = mock_llm.generate_response.call_args[0][0]
        assert user_input in call_args
        assert "!" in call_args
        assert "?" in call_args
        assert "&" in call_args

    @patch('chat_interface.llm')
    def test_chat_response_with_unicode_characters(self, mock_llm):
        """Test chat response with unicode characters"""
        mock_llm.generate_response.return_value = "International communication feedback"
        
        user_input = "Bonjour! Café résumé naïve 你好 🎉"
        result = chat_response(user_input)
        
        assert isinstance(result, str)
        assert result == "International communication feedback"
        
        # Verify unicode characters were preserved
        call_args = mock_llm.generate_response.call_args[0][0]
        assert user_input in call_args

    @patch('chat_interface.llm')
    def test_chat_response_multiple_calls_independence(self, mock_llm):
        """Test that multiple chat responses are independent"""
        mock_llm.generate_response.side_effect = [
            "First response",
            "Second response", 
            "Third response"
        ]
        
        # Make multiple calls with different inputs
        result1 = chat_response("First message")
        result2 = chat_response("Second message")
        result3 = chat_response("Third message")
        
        assert result1 == "First response"
        assert result2 == "Second response"
        assert result3 == "Third response"
        
        # Verify each call was made with correct input
        calls = mock_llm.generate_response.call_args_list
        assert len(calls) == 3
        assert "First message" in calls[0][0][0]
        assert "Second message" in calls[1][0][0]
        assert "Third message" in calls[2][0][0]

    @patch('chat_interface.llm')
    def test_chat_response_error_handling(self, mock_llm):
        """Test error handling when LLM fails"""
        mock_llm.generate_response.side_effect = Exception("LLM generation failed")
        
        with pytest.raises(Exception, match="LLM generation failed"):
            chat_response("Test input")

    @patch('chat_interface.llm')
    def test_chat_response_conversational_coaching_scenarios(self, mock_llm):
        """Test various conversational coaching scenarios"""
        test_scenarios = [
            {
                "input": "I get nervous during presentations",
                "expected_feedback": "Focus on breathing techniques and practice to build confidence."
            },
            {
                "input": "How can I improve my tone of voice?",
                "expected_feedback": "Practice varying your pitch and pace for better engagement."
            },
            {
                "input": "I struggle with eye contact",
                "expected_feedback": "Start by looking at friendly faces and gradually expand your gaze."
            }
        ]
        
        for scenario in test_scenarios:
            mock_llm.generate_response.return_value = scenario["expected_feedback"]
            result = chat_response(scenario["input"])
            
            assert isinstance(result, str)
            assert result == scenario["expected_feedback"]
            
            # Verify coaching context is maintained in prompt
            call_args = mock_llm.generate_response.call_args[0][0]
            assert "communication expert" in call_args
            assert scenario["input"] in call_args

    @patch('chat_interface.llm')
    def test_chat_response_feedback_categories(self, mock_llm):
        """Test that feedback covers all required categories"""
        mock_llm.generate_response.return_value = "Comprehensive feedback covering all areas"
        
        user_input = "I want to improve my communication skills"
        chat_response(user_input)
        
        call_args = mock_llm.generate_response.call_args[0][0]
        
        # Verify all feedback categories are mentioned in prompt
        required_categories = ["clarity", "tone", "suggestions for improvement"]
        for category in required_categories:
            assert category in call_args

    @patch('chat_interface.llm')
    def test_chat_response_return_type_consistency(self, mock_llm):
        """Test that chat response always returns a string"""
        test_responses = [
            "Normal response",
            "",  # Empty string
            "   ",  # Whitespace string
            "Multi-line\nresponse\nwith\nbreaks"
        ]
        
        for expected_response in test_responses:
            mock_llm.generate_response.return_value = expected_response
            result = chat_response("Test input")
            
            assert isinstance(result, str)
            assert result == expected_response

    def test_chat_response_function_signature(self):
        """Test that the function has the expected signature"""
        import inspect
        
        sig = inspect.signature(chat_response)
        params = list(sig.parameters.keys())
        
        assert len(params) == 1
        assert params[0] == "user_input"
        assert callable(chat_response)

    @patch('chat_interface.LLMWrapper')
    def test_module_level_llm_initialization(self, mock_llm_class):
        """Test that LLMWrapper is initialized at module level"""
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        
        # Reload module to trigger initialization
        import importlib
        importlib.reload(chat_interface)
        
        # Verify LLMWrapper was instantiated
        mock_llm_class.assert_called_once()

    @patch('chat_interface.gr.Interface')
    def test_gradio_interface_configuration(self, mock_interface):
        """Test that Gradio interface is properly configured"""
        mock_interface_instance = Mock()
        mock_interface.return_value = mock_interface_instance
        
        # Reload module to trigger interface creation
        import importlib
        importlib.reload(chat_interface)
        
        # Verify Gradio interface was created with correct parameters
        mock_interface.assert_called_once()
        call_kwargs = mock_interface.call_args[1]
        
        assert call_kwargs['fn'] == chat_interface.chat_response
        assert call_kwargs['outputs'] == "text"
        assert "Verbal Communication Skills Trainer - Chat" in call_kwargs['title']
        assert "Enter your message to receive expert communication feedback" in call_kwargs['description']

    @patch('chat_interface.gr.Textbox')
    def test_gradio_input_configuration(self, mock_textbox):
        """Test that Gradio input textbox is properly configured"""
        mock_textbox_instance = Mock()
        mock_textbox.return_value = mock_textbox_instance
        
        # Reload module to trigger interface creation
        import importlib
        importlib.reload(chat_interface)
        
        # Verify textbox was configured correctly
        mock_textbox.assert_called_once()
        call_kwargs = mock_textbox.call_args[1]
        
        assert call_kwargs['lines'] == 4
        assert "Enter your message here..." in call_kwargs['placeholder']

    @patch('chat_interface.llm')
    def test_message_handling_edge_cases(self, mock_llm):
        """Test message handling with various edge cases"""
        edge_cases = [
            "\n\n\n",  # Only newlines
            "a",  # Single character
            "A" * 10000,  # Very long single word
            "🎉" * 100,  # Many emojis
            "123456789",  # Only numbers
            "!@#$%^&*()",  # Only special characters
        ]
        
        for i, edge_case in enumerate(edge_cases):
            mock_llm.generate_response.return_value = f"Response {i}"
            result = chat_response(edge_case)
            
            assert isinstance(result, str)
            assert result == f"Response {i}"
            
            # Verify edge case was properly included in prompt
            call_args = mock_llm.generate_response.call_args[0][0]
            assert edge_case in call_args

    @patch('chat_interface.llm')
    def test_session_management_simulation(self, mock_llm):
        """Test simulated session management through multiple interactions"""
        # Simulate a conversation session with multiple exchanges
        conversation_flow = [
            ("Hello, I need help with public speaking", "Welcome! I'd be happy to help with public speaking."),
            ("I get very nervous before presentations", "Nervousness is normal. Let's work on confidence building."),
            ("What breathing techniques do you recommend?", "Try the 4-7-8 breathing technique before speaking."),
            ("Thank you, that's very helpful!", "You're welcome! Keep practicing these techniques.")
        ]
        
        for user_msg, expected_response in conversation_flow:
            mock_llm.generate_response.return_value = expected_response
            result = chat_response(user_msg)
            
            assert isinstance(result, str)
            assert result == expected_response
            
            # Verify each message maintains coaching context
            call_args = mock_llm.generate_response.call_args[0][0]
            assert "communication expert" in call_args
            assert user_msg in call_args

    @patch('chat_interface.llm')
    def test_response_generation_consistency(self, mock_llm):
        """Test that response generation is consistent and reliable"""
        mock_llm.generate_response.return_value = "Consistent coaching response"
        
        # Test same input multiple times
        user_input = "I want to improve my communication"
        results = []
        
        for _ in range(5):
            result = chat_response(user_input)
            results.append(result)
        
        # All results should be identical (since we're mocking)
        assert all(result == "Consistent coaching response" for result in results)
        assert len(set(results)) == 1  # All results are the same
        
        # Verify LLM was called each time
        assert mock_llm.generate_response.call_count == 5

    @patch('chat_interface.llm')
    def test_text_based_coaching_features(self, mock_llm):
        """Test specific text-based conversational coaching features"""
        coaching_scenarios = [
            {
                "category": "clarity",
                "input": "I mumble when I speak",
                "response": "Focus on articulation exercises and speaking more slowly."
            },
            {
                "category": "tone",
                "input": "People say I sound monotone",
                "response": "Practice varying your pitch and adding emotional inflection."
            },
            {
                "category": "structure",
                "input": "My presentations are disorganized",
                "response": "Use the introduction-body-conclusion structure with clear transitions."
            }
        ]
        
        for scenario in coaching_scenarios:
            mock_llm.generate_response.return_value = scenario["response"]
            result = chat_response(scenario["input"])
            
            assert isinstance(result, str)
            assert result == scenario["response"]
            
            # Verify coaching expertise is emphasized in prompt
            call_args = mock_llm.generate_response.call_args[0][0]
            assert "communication expert" in call_args
            assert "detailed feedback" in call_args
            assert scenario["input"] in call_args
