import pytest
import json
import random
from unittest.mock import Mock, patch, mock_open
import sys
sys.path.append('/home/daytona/skill-training-llm/verbal_comm_skills_trainer')
import training_modules
from training_modules import (
    get_impromptu_prompt, 
    get_storytelling_prompt, 
    get_conflict_resolution_prompt,
    evaluate_training
)


class TestTrainingModules:
    """Test suite for training modules functionality"""

    @pytest.fixture
    def mock_config_data(self):
        """Mock configuration data for testing"""
        return {
            "model_name": "test-model",
            "quantization": True,
            "impromptu_prompts": [
                "Why is empathy important?",
                "What motivates you in difficult situations?",
                "Discuss the importance of effective teamwork."
            ],
            "storytelling_prompts": [
                "Tell a story about a time you overcame a challenge.",
                "Share an experience where you learned a valuable lesson.",
                "Describe a moment that changed your perspective."
            ],
            "conflict_resolution_prompts": [
                "Your teammate is frustrated due to workload imbalance. How do you address it?",
                "You have a conflict with a colleague over project direction. What do you do?",
                "A client is unhappy with your team's deliverable. How do you handle it?"
            ]
        }

    @pytest.fixture
    def mock_llm_wrapper(self):
        """Mock LLMWrapper for testing evaluation functions"""
        mock_llm = Mock()
        mock_llm.generate_response.return_value = "Detailed feedback on communication skills with suggestions for improvement."
        return mock_llm

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_config_loading(self, mock_json_load, mock_file, mock_config_data):
        """Test that configuration is loaded correctly"""
        mock_json_load.return_value = mock_config_data
        
        # Reload the module to trigger config loading
        import importlib
        importlib.reload(training_modules)
        
        mock_file.assert_called_with("config.json", "r")
        mock_json_load.assert_called_once()

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('random.choice')
    def test_get_impromptu_prompt_success(self, mock_choice, mock_json_load, mock_file, mock_config_data):
        """Test successful retrieval of impromptu speaking prompt"""
        mock_json_load.return_value = mock_config_data
        mock_choice.return_value = "Why is empathy important?"
        
        # Reload module to use mocked config
        import importlib
        importlib.reload(training_modules)
        
        prompt = training_modules.get_impromptu_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert prompt == "Why is empathy important?"
        mock_choice.assert_called_once_with(mock_config_data["impromptu_prompts"])

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('random.choice')
    def test_get_storytelling_prompt_success(self, mock_choice, mock_json_load, mock_file, mock_config_data):
        """Test successful retrieval of storytelling prompt"""
        mock_json_load.return_value = mock_config_data
        mock_choice.return_value = "Tell a story about a time you overcame a challenge."
        
        # Reload module to use mocked config
        import importlib
        importlib.reload(training_modules)
        
        prompt = training_modules.get_storytelling_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert prompt == "Tell a story about a time you overcame a challenge."
        mock_choice.assert_called_once_with(mock_config_data["storytelling_prompts"])

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('random.choice')
    def test_get_conflict_resolution_prompt_success(self, mock_choice, mock_json_load, mock_file, mock_config_data):
        """Test successful retrieval of conflict resolution prompt"""
        mock_json_load.return_value = mock_config_data
        mock_choice.return_value = "Your teammate is frustrated due to workload imbalance. How do you address it?"
        
        # Reload module to use mocked config
        import importlib
        importlib.reload(training_modules)
        
        prompt = training_modules.get_conflict_resolution_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert prompt == "Your teammate is frustrated due to workload imbalance. How do you address it?"
        mock_choice.assert_called_once_with(mock_config_data["conflict_resolution_prompts"])

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('random.choice')
    def test_prompt_functions_with_empty_config(self, mock_choice, mock_json_load, mock_file):
        """Test prompt functions when config has empty prompt lists"""
        empty_config = {
            "impromptu_prompts": [],
            "storytelling_prompts": [],
            "conflict_resolution_prompts": []
        }
        mock_json_load.return_value = empty_config
        mock_choice.side_effect = IndexError("list index out of range")
        
        # Reload module to use mocked config
        import importlib
        importlib.reload(training_modules)
        
        with pytest.raises(IndexError):
            training_modules.get_impromptu_prompt()

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('random.choice')
    def test_prompt_functions_with_missing_keys(self, mock_choice, mock_json_load, mock_file):
        """Test prompt functions when config is missing required keys"""
        incomplete_config = {"model_name": "test-model"}
        mock_json_load.return_value = incomplete_config
        mock_choice.return_value = []  # Empty list from config.get() default
        
        # Reload module to use mocked config
        import importlib
        importlib.reload(training_modules)
        
        # Should handle missing keys gracefully by returning empty list default
        with pytest.raises(IndexError):
            training_modules.get_impromptu_prompt()

    @patch('training_modules.llm')
    def test_evaluate_training_impromptu_module(self, mock_llm):
        """Test evaluation for impromptu speaking module"""
        mock_llm.generate_response.return_value = "Good structure and clarity. Consider adding more specific examples."
        
        user_response = "I believe empathy is crucial because it helps us understand others' perspectives."
        feedback = evaluate_training("impromptu", user_response)
        
        assert isinstance(feedback, str)
        assert len(feedback) > 0
        
        # Verify the correct prompt was constructed
        expected_prompt = (
            "You are a communication expert. Critique the structure, fluency, and clarity of this impromptu speech. "
            "Here is the user's response: " + user_response
        )
        mock_llm.generate_response.assert_called_once_with(expected_prompt)

    @patch('training_modules.llm')
    def test_evaluate_training_storytelling_module(self, mock_llm):
        """Test evaluation for storytelling module"""
        mock_llm.generate_response.return_value = "Engaging narrative with good character development. The pacing could be improved."
        
        user_response = "Once upon a time, I faced a difficult challenge at work that taught me resilience."
        feedback = evaluate_training("storytelling", user_response)
        
        assert isinstance(feedback, str)
        assert len(feedback) > 0
        
        # Verify the correct prompt was constructed
        expected_prompt = (
            "You are a communication expert. Evaluate the narrative quality and engagement of the story. "
            "Here is the user's response: " + user_response
        )
        mock_llm.generate_response.assert_called_once_with(expected_prompt)

    @patch('training_modules.llm')
    def test_evaluate_training_conflict_resolution_module(self, mock_llm):
        """Test evaluation for conflict resolution module"""
        mock_llm.generate_response.return_value = "Diplomatic approach with good listening skills. Consider more specific action steps."
        
        user_response = "I would first listen to my teammate's concerns and then work together to find a solution."
        feedback = evaluate_training("conflict", user_response)
        
        assert isinstance(feedback, str)
        assert len(feedback) > 0
        
        # Verify the correct prompt was constructed
        expected_prompt = (
            "You are a communication expert. Assess the diplomatic and effective handling of the conflict scenario. "
            "Here is the user's response: " + user_response
        )
        mock_llm.generate_response.assert_called_once_with(expected_prompt)

    @patch('training_modules.llm')
    def test_evaluate_training_unknown_module(self, mock_llm):
        """Test evaluation with unknown module type"""
        user_response = "Some response text"
        feedback = evaluate_training("unknown_module", user_response)
        
        assert feedback == "Unknown module."
        # LLM should not be called for unknown modules
        mock_llm.generate_response.assert_not_called()

    @patch('training_modules.llm')
    def test_evaluate_training_empty_response(self, mock_llm):
        """Test evaluation with empty user response"""
        mock_llm.generate_response.return_value = "Please provide a response to evaluate."
        
        feedback = evaluate_training("impromptu", "")
        
        assert isinstance(feedback, str)
        expected_prompt = (
            "You are a communication expert. Critique the structure, fluency, and clarity of this impromptu speech. "
            "Here is the user's response: "
        )
        mock_llm.generate_response.assert_called_once_with(expected_prompt)

    @patch('training_modules.llm')
    def test_evaluate_training_llm_error_handling(self, mock_llm):
        """Test evaluation when LLM generates an error"""
        mock_llm.generate_response.side_effect = Exception("LLM generation failed")
        
        with pytest.raises(Exception, match="LLM generation failed"):
            evaluate_training("impromptu", "Test response")

    @patch('training_modules.llm')
    def test_evaluate_training_all_modules_with_same_response(self, mock_llm):
        """Test that different modules generate different evaluation prompts"""
        mock_llm.generate_response.return_value = "Generic feedback"
        user_response = "This is my response to the training prompt."
        
        # Test all three modules
        evaluate_training("impromptu", user_response)
        evaluate_training("storytelling", user_response)
        evaluate_training("conflict", user_response)
        
        # Verify LLM was called three times with different prompts
        assert mock_llm.generate_response.call_count == 3
        
        call_args_list = mock_llm.generate_response.call_args_list
        prompts = [call[0][0] for call in call_args_list]
        
        # Verify each prompt contains the appropriate instruction
        assert "structure, fluency, and clarity" in prompts[0]  # impromptu
        assert "narrative quality and engagement" in prompts[1]  # storytelling
        assert "diplomatic and effective handling" in prompts[2]  # conflict

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_module_initialization_with_llm_wrapper(self, mock_json_load, mock_file, mock_config_data):
        """Test that the module properly initializes LLMWrapper"""
        mock_json_load.return_value = mock_config_data
        
        # Reload module to trigger LLMWrapper initialization
        with patch('training_modules.LLMWrapper') as mock_llm_class:
            mock_llm_instance = Mock()
            mock_llm_class.return_value = mock_llm_instance
            
            import importlib
            importlib.reload(training_modules)
            
            # Verify LLMWrapper was instantiated
            mock_llm_class.assert_called_once()

    def test_module_constants_and_structure(self):
        """Test that the module has the expected structure and constants"""
        # Verify all required functions are available
        assert hasattr(training_modules, 'get_impromptu_prompt')
        assert hasattr(training_modules, 'get_storytelling_prompt')
        assert hasattr(training_modules, 'get_conflict_resolution_prompt')
        assert hasattr(training_modules, 'evaluate_training')
        
        # Verify functions are callable
        assert callable(training_modules.get_impromptu_prompt)
        assert callable(training_modules.get_storytelling_prompt)
        assert callable(training_modules.get_conflict_resolution_prompt)
        assert callable(training_modules.evaluate_training)

    @patch('training_modules.llm')
    def test_evaluate_training_case_sensitivity(self, mock_llm):
        """Test that module names are handled correctly regardless of case"""
        mock_llm.generate_response.return_value = "Test feedback"
        
        # Test different case variations
        feedback1 = evaluate_training("IMPROMPTU", "test response")
        feedback2 = evaluate_training("Storytelling", "test response")
        feedback3 = evaluate_training("CONFLICT", "test response")
        
        # All should return "Unknown module." since exact case matching is expected
        assert feedback1 == "Unknown module."
        assert feedback2 == "Unknown module."
        assert feedback3 == "Unknown module."
        
        # Only exact matches should work
        feedback4 = evaluate_training("impromptu", "test response")
        assert feedback4 != "Unknown module."

