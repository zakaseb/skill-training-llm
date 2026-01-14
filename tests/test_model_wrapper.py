import pytest
import json
import os
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('/home/daytona/skill-training-llm/verbal_comm_skills_trainer')
from model_wrapper import LLMWrapper


class TestLLMWrapper:
    """Test suite for LLMWrapper class"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a temporary config file for testing"""
        config_data = {
            "model_name": "test-model",
            "quantization": True
        }
        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)
        return str(config_file)

    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing"""
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.decode.return_value = "Mocked response text"
        return tokenizer

    @pytest.fixture
    def mock_model(self):
        """Mock model for testing"""
        model = Mock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        return model

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.is_available')
    def test_init_with_cuda_available(self, mock_cuda, mock_model_class, mock_tokenizer_class, mock_config):
        """Test LLMWrapper initialization when CUDA is available"""
        mock_cuda.return_value = True
        mock_tokenizer_class.return_value = Mock()
        mock_model_class.return_value = Mock()
        
        llm = LLMWrapper(config_path=mock_config)
        
        assert llm.model_name == "test-model"
        assert llm.quantization is True
        assert llm.device == "cuda"
        mock_tokenizer_class.assert_called_once_with("test-model")
        mock_model_class.assert_called_once()

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.is_available')
    def test_init_with_cuda_unavailable(self, mock_cuda, mock_model_class, mock_tokenizer_class, mock_config):
        """Test LLMWrapper initialization when CUDA is not available"""
        mock_cuda.return_value = False
        mock_tokenizer_class.return_value = Mock()
        mock_model_class.return_value = Mock()
        
        llm = LLMWrapper(config_path=mock_config)
        
        assert llm.device == "cpu"

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_init_with_default_config(self, mock_model_class, mock_tokenizer_class):
        """Test LLMWrapper initialization with default config values"""
        mock_tokenizer_class.return_value = Mock()
        mock_model_class.return_value = Mock()
        
        # Create a minimal config file
        with patch('builtins.open', mock_open_config({})):
            llm = LLMWrapper()
            
        assert llm.model_name == "mistralai/Mistral-7B"  # default value
        assert llm.quantization is True  # default value

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test successful model loading"""
        mock_tokenizer_class.return_value = Mock()
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        
        mock_model_class.assert_called_once_with(
            "test-model",
            device_map="auto",
            load_in_4bit=True,
            trust_remote_code=True
        )
        assert llm.model == mock_model

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_load_model_failure(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test model loading failure handling"""
        mock_tokenizer_class.return_value = Mock()
        mock_model_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            LLMWrapper(config_path=mock_config)

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.is_available')
    def test_generate_response_success(self, mock_cuda, mock_model_class, mock_tokenizer_class, mock_config):
        """Test successful response generation"""
        mock_cuda.return_value = False
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Test prompt: Provide a brief feedback. Generated response."
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        response = llm.generate_response("Test prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0
        mock_tokenizer.assert_called_with("Test prompt", return_tensors="pt")
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once_with(torch.tensor([1, 2, 3, 4, 5]), skip_special_tokens=True)

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.is_available')
    def test_generate_response_with_custom_parameters(self, mock_cuda, mock_model_class, mock_tokenizer_class, mock_config):
        """Test response generation with custom parameters"""
        mock_cuda.return_value = False
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Custom response"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        response = llm.generate_response("Test prompt", max_length=500, temperature=0.9)
        
        # Verify custom parameters were passed to model.generate
        call_args = mock_model.generate.call_args
        assert call_args[1]['max_length'] == 500
        assert call_args[1]['temperature'] == 0.9
        assert call_args[1]['do_sample'] is True
        assert call_args[1]['num_return_sequences'] == 1

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_generate_response_error_handling(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test error handling during response generation"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.side_effect = Exception("Generation failed")
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        response = llm.generate_response("Test prompt")
        
        assert response == "Error generating response."

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_caching_mechanism(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test that caching works for identical prompts"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "Cached response"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        
        # First call
        response1 = llm.generate_response("Same prompt")
        # Second call with same prompt
        response2 = llm.generate_response("Same prompt")
        
        assert response1 == response2
        # Model.generate should only be called once due to caching
        assert mock_model.generate.call_count == 1

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_batch_generate(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test batch generation functionality"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.side_effect = ["Response 1", "Response 2", "Response 3"]
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = llm.batch_generate(prompts)
        
        assert len(responses) == 3
        assert all(isinstance(response, str) for response in responses)
        assert mock_model.generate.call_count == 3

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_batch_generate_with_custom_parameters(self, mock_model_class, mock_tokenizer_class, mock_config):
        """Test batch generation with custom parameters"""
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.side_effect = ["Response 1", "Response 2"]
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_class.return_value = mock_model
        
        llm = LLMWrapper(config_path=mock_config)
        prompts = ["Prompt 1", "Prompt 2"]
        responses = llm.batch_generate(prompts, max_length=300, temperature=0.8)
        
        assert len(responses) == 2
        # Verify parameters were passed correctly
        for call in mock_model.generate.call_args_list:
            assert call[1]['max_length'] == 300
            assert call[1]['temperature'] == 0.8

    def test_config_file_not_found(self):
        """Test handling of missing config file"""
        with pytest.raises(FileNotFoundError):
            LLMWrapper(config_path="nonexistent_config.json")

    @patch('model_wrapper.AutoTokenizer.from_pretrained')
    @patch('model_wrapper.AutoModelForCausalLM.from_pretrained')
    def test_invalid_config_format(self, mock_model_class, mock_tokenizer_class, tmp_path):
        """Test handling of invalid config file format"""
        # Create invalid JSON config
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, "w") as f:
            f.write("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            LLMWrapper(config_path=str(config_file))


def mock_open_config(config_data):
    """Helper function to mock config file opening"""
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(config_data))

