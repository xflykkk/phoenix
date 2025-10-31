"""
Test suite for playground_models query.
Tests custom model retrieval from database (not from registry).
"""

import re
from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy import insert

from phoenix.db import models
from phoenix.server.api.types.GenerativeProvider import GenerativeProviderKey
from phoenix.server.types import DbSessionFactory
from tests.unit.graphql import AsyncGraphQLClient


@pytest.fixture
async def custom_models(db: DbSessionFactory) -> None:
    """Create test custom models in the database."""
    async with db() as session:
        # Create custom OpenAI models
        openai_model_1 = models.GenerativeModel(
            name="gpt-4o-custom",
            provider="OPENAI",
            name_pattern=re.compile("gpt-4o-custom"),
            is_built_in=False,
        )
        openai_model_2 = models.GenerativeModel(
            name="gpt-4-turbo-custom",
            provider="OPENAI",
            name_pattern=re.compile("gpt-4-turbo-custom"),
            is_built_in=False,
        )

        # Create custom Anthropic model
        anthropic_model = models.GenerativeModel(
            name="claude-3-5-sonnet-custom",
            provider="ANTHROPIC",
            name_pattern=re.compile("claude-3-5-sonnet-custom"),
            is_built_in=False,
        )

        # Create a built-in model (should NOT appear in results)
        builtin_model = models.GenerativeModel(
            name="gpt-4-builtin",
            provider="OPENAI",
            name_pattern=re.compile("gpt-4"),
            is_built_in=True,
        )

        # Create a deleted model (should NOT appear in results)
        deleted_model = models.GenerativeModel(
            name="gpt-4-deleted",
            provider="OPENAI",
            name_pattern=re.compile("gpt-4-deleted"),
            is_built_in=False,
            deleted_at=datetime.now(timezone.utc),
        )

        session.add_all(
            [
                openai_model_1,
                openai_model_2,
                anthropic_model,
                builtin_model,
                deleted_model,
            ]
        )

        # Add token prices for custom models
        for model in [openai_model_1, openai_model_2, anthropic_model]:
            session.add(
                models.TokenPrice(
                    model=model,
                    token_type="input",
                    is_prompt=True,
                    base_rate=0.000005,  # $5 per million tokens
                )
            )
            session.add(
                models.TokenPrice(
                    model=model,
                    token_type="output",
                    is_prompt=False,
                    base_rate=0.000015,  # $15 per million tokens
                )
            )

        await session.commit()


class TestPlaygroundModels:
    """Test cases for playground_models query."""

    async def test_returns_only_custom_models(
        self,
        gql_client: AsyncGraphQLClient,
        custom_models: Any,
    ) -> None:
        """Test that only custom models (not built-in) are returned."""
        query = """
          query {
            playgroundModels {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(query=query)
        assert not response.errors

        models = response.data["playgroundModels"]
        model_names = [m["name"] for m in models]

        # Should include custom models
        assert "gpt-4o-custom" in model_names
        assert "gpt-4-turbo-custom" in model_names
        assert "claude-3-5-sonnet-custom" in model_names

        # Should NOT include built-in model
        assert "gpt-4-builtin" not in model_names

        # Should NOT include deleted model
        assert "gpt-4-deleted" not in model_names

        # Total should be 3 (only custom, not deleted models)
        assert len(models) == 3

    async def test_filter_by_openai_provider(
        self,
        gql_client: AsyncGraphQLClient,
        custom_models: Any,
    ) -> None:
        """Test filtering models by OpenAI provider."""
        query = """
          query ($input: ModelsInput) {
            playgroundModels(input: $input) {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(
            query=query, variables={"input": {"providerKey": "OPENAI"}}
        )
        assert not response.errors

        models = response.data["playgroundModels"]
        model_names = [m["name"] for m in models]

        # Should only include OpenAI custom models
        assert "gpt-4o-custom" in model_names
        assert "gpt-4-turbo-custom" in model_names

        # Should NOT include Anthropic model
        assert "claude-3-5-sonnet-custom" not in model_names

        # All models should have OPENAI provider
        for model in models:
            assert model["providerKey"] == "OPENAI"

        assert len(models) == 2

    async def test_filter_by_anthropic_provider(
        self,
        gql_client: AsyncGraphQLClient,
        custom_models: Any,
    ) -> None:
        """Test filtering models by Anthropic provider."""
        query = """
          query ($input: ModelsInput) {
            playgroundModels(input: $input) {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(
            query=query, variables={"input": {"providerKey": "ANTHROPIC"}}
        )
        assert not response.errors

        models = response.data["playgroundModels"]
        model_names = [m["name"] for m in models]

        # Should only include Anthropic custom model
        assert "claude-3-5-sonnet-custom" in model_names

        # Should NOT include OpenAI models
        assert "gpt-4o-custom" not in model_names
        assert "gpt-4-turbo-custom" not in model_names

        # All models should have ANTHROPIC provider
        for model in models:
            assert model["providerKey"] == "ANTHROPIC"

        assert len(models) == 1

    async def test_empty_result_when_no_custom_models(
        self,
        gql_client: AsyncGraphQLClient,
        db: DbSessionFactory,
    ) -> None:
        """Test that empty list is returned when no custom models exist."""
        query = """
          query {
            playgroundModels {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(query=query)
        assert not response.errors

        models = response.data["playgroundModels"]
        # Should return empty list when no custom models
        assert len(models) == 0

    async def test_excludes_models_without_provider(
        self,
        gql_client: AsyncGraphQLClient,
        db: DbSessionFactory,
    ) -> None:
        """Test that models without a provider are excluded."""
        async with db() as session:
            # Create a custom model without provider
            no_provider_model = models.GenerativeModel(
                name="no-provider-model",
                provider="",  # Empty provider
                name_pattern=re.compile("no-provider-model"),
                is_built_in=False,
            )
            session.add(no_provider_model)
            await session.commit()

        query = """
          query {
            playgroundModels {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(query=query)
        assert not response.errors

        models = response.data["playgroundModels"]
        model_names = [m["name"] for m in models]

        # Should NOT include model without provider
        assert "no-provider-model" not in model_names

    async def test_handles_invalid_provider_gracefully(
        self,
        gql_client: AsyncGraphQLClient,
        db: DbSessionFactory,
    ) -> None:
        """Test that models with invalid provider values are skipped."""
        async with db() as session:
            # Create a model with invalid provider
            invalid_provider_model = models.GenerativeModel(
                name="invalid-provider-model",
                provider="INVALID_PROVIDER",  # Not a valid GenerativeProviderKey
                name_pattern=re.compile("invalid-provider-model"),
                is_built_in=False,
            )
            session.add(invalid_provider_model)
            await session.commit()

        query = """
          query {
            playgroundModels {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(query=query)
        assert not response.errors

        models = response.data["playgroundModels"]
        model_names = [m["name"] for m in models]

        # Should NOT include model with invalid provider
        assert "invalid-provider-model" not in model_names

    async def test_model_fields_are_correct(
        self,
        gql_client: AsyncGraphQLClient,
        custom_models: Any,
    ) -> None:
        """Test that returned models have correct field structure."""
        query = """
          query {
            playgroundModels {
              name
              providerKey
            }
          }
        """
        response = await gql_client.execute(query=query)
        assert not response.errors

        models = response.data["playgroundModels"]
        assert len(models) > 0

        # Verify structure of first model
        model = models[0]
        assert "name" in model
        assert "providerKey" in model
        assert isinstance(model["name"], str)
        assert isinstance(model["providerKey"], str)

        # Verify provider key is valid
        valid_providers = [
            "OPENAI",
            "ANTHROPIC",
            "AZURE_OPENAI",
            "GOOGLE",
            "AWS",
            "DEEPSEEK",
            "XAI",
            "OLLAMA",
        ]
        assert model["providerKey"] in valid_providers
