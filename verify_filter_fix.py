#!/usr/bin/env python3
"""
验证完整的 provider 过滤和转换逻辑
模拟前端查询 → 后端过滤 → 数据库查询 → 结果转换的完整流程
"""

# 模拟 OpenInferenceLLMProviderValues
class OpenInferenceLLMProviderValues:
    OPENAI = type('obj', (object,), {'value': 'openai'})()
    ANTHROPIC = type('obj', (object,), {'value': 'anthropic'})()
    AZURE = type('obj', (object,), {'value': 'azure'})()
    GOOGLE = type('obj', (object,), {'value': 'google'})()
    AWS = type('obj', (object,), {'value': 'aws'})()
    DEEPSEEK = type('obj', (object,), {'value': 'deepseek'})()
    XAI = type('obj', (object,), {'value': 'xai'})()

# 模拟 GenerativeProviderKey enum
class GenerativeProviderKey:
    class Key:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def __eq__(self, other):
            return isinstance(other, GenerativeProviderKey.Key) and self.name == other.name

        def __repr__(self):
            return f"GenerativeProviderKey.{self.name}"

    OPENAI = Key("OPENAI", "OpenAI")
    ANTHROPIC = Key("ANTHROPIC", "Anthropic")
    AZURE_OPENAI = Key("AZURE_OPENAI", "Azure OpenAI")
    GOOGLE = Key("GOOGLE", "Google AI Studio")
    DEEPSEEK = Key("DEEPSEEK", "DeepSeek")
    XAI = Key("XAI", "xAI")
    AWS = Key("AWS", "AWS Bedrock")

# 反向转换: GenerativeProviderKey → semconv string
def _gql_generative_provider_key_to_semconv_provider(provider_key):
    """将 GQL GenerativeProviderKey 转换为 semconv provider 字符串(小写格式)"""
    if provider_key == GenerativeProviderKey.OPENAI:
        return OpenInferenceLLMProviderValues.OPENAI.value
    if provider_key == GenerativeProviderKey.ANTHROPIC:
        return OpenInferenceLLMProviderValues.ANTHROPIC.value
    if provider_key == GenerativeProviderKey.AZURE_OPENAI:
        return OpenInferenceLLMProviderValues.AZURE.value
    if provider_key == GenerativeProviderKey.GOOGLE:
        return OpenInferenceLLMProviderValues.GOOGLE.value
    if provider_key == GenerativeProviderKey.DEEPSEEK:
        return OpenInferenceLLMProviderValues.DEEPSEEK.value
    if provider_key == GenerativeProviderKey.XAI:
        return OpenInferenceLLMProviderValues.XAI.value
    if provider_key == GenerativeProviderKey.AWS:
        return OpenInferenceLLMProviderValues.AWS.value
    return None

# 正向转换: semconv string → GenerativeProviderKey
def _semconv_provider_to_gql_generative_provider_key(semconv_provider_str):
    """将 semconv provider 字符串转换为 GQL GenerativeProviderKey"""
    if semconv_provider_str == "openai":
        return GenerativeProviderKey.OPENAI
    if semconv_provider_str == "anthropic":
        return GenerativeProviderKey.ANTHROPIC
    if semconv_provider_str == "azure":
        return GenerativeProviderKey.AZURE_OPENAI
    if semconv_provider_str == "google":
        return GenerativeProviderKey.GOOGLE
    if semconv_provider_str == "deepseek":
        return GenerativeProviderKey.DEEPSEEK
    if semconv_provider_str == "xai":
        return GenerativeProviderKey.XAI
    if semconv_provider_str == "aws":
        return GenerativeProviderKey.AWS
    return None

# 模拟数据库中的记录
database_records = [
    {"name": "gpt-4o-custom", "provider": "openai"},
    {"name": "gpt-4-turbo-custom", "provider": "openai"},
    {"name": "claude-3-5-sonnet-custom", "provider": "anthropic"},
    {"name": "deepseek-coder", "provider": "deepseek"},
]

print("=" * 70)
print("验证完整的 Provider 过滤和转换流程")
print("=" * 70)
print()

# 测试场景
test_scenarios = [
    {
        "name": "查询 OPENAI 模型",
        "frontend_provider": GenerativeProviderKey.OPENAI,
        "expected_models": ["gpt-4o-custom", "gpt-4-turbo-custom"],
    },
    {
        "name": "查询 ANTHROPIC 模型",
        "frontend_provider": GenerativeProviderKey.ANTHROPIC,
        "expected_models": ["claude-3-5-sonnet-custom"],
    },
    {
        "name": "查询 DEEPSEEK 模型",
        "frontend_provider": GenerativeProviderKey.DEEPSEEK,
        "expected_models": ["deepseek-coder"],
    },
]

all_passed = True

for scenario in test_scenarios:
    print(f"📝 测试场景: {scenario['name']}")
    print("-" * 70)

    # 步骤 1: 前端发送 GenerativeProviderKey
    frontend_provider = scenario["frontend_provider"]
    print(f"  1️⃣ 前端发送: {frontend_provider}")
    print(f"     枚举显示值: {frontend_provider.value}")

    # 步骤 2: 后端转换为 semconv 格式用于数据库查询
    semconv_filter = _gql_generative_provider_key_to_semconv_provider(frontend_provider)
    print(f"  2️⃣ 转换为数据库过滤条件: provider = '{semconv_filter}'")

    # 步骤 3: 模拟数据库查询
    filtered_records = [
        record for record in database_records
        if record["provider"] == semconv_filter
    ]
    print(f"  3️⃣ 数据库查询结果: {len(filtered_records)} 条记录")
    for record in filtered_records:
        print(f"     - {record['name']} (provider='{record['provider']}')")

    # 步骤 4: 转换回 GenerativeProviderKey 返回前端
    results = []
    for record in filtered_records:
        provider_key = _semconv_provider_to_gql_generative_provider_key(record["provider"])
        if provider_key:
            results.append({
                "name": record["name"],
                "providerKey": provider_key
            })

    print(f"  4️⃣ 返回前端结果: {len(results)} 个模型")
    for result in results:
        print(f"     - {result['name']} (providerKey={result['providerKey']})")

    # 验证
    result_model_names = [r["name"] for r in results]
    expected_models = scenario["expected_models"]

    if set(result_model_names) == set(expected_models):
        print(f"  ✅ 测试通过!")
    else:
        print(f"  ❌ 测试失败!")
        print(f"     期望: {expected_models}")
        print(f"     实际: {result_model_names}")
        all_passed = False

    print()

print("=" * 70)
if all_passed:
    print("✅ 所有测试通过!")
    print()
    print("修复总结:")
    print("  问题1 ✅ 转换函数: 使用 _semconv_provider_to_gql_generative_provider_key()")
    print("  问题2 ✅ 过滤逻辑: 添加 _gql_generative_provider_key_to_semconv_provider()")
    print("            将 GenerativeProviderKey 转换为小写 semconv 格式进行数据库查询")
    print()
    print("数据流:")
    print("  前端: OPENAI enum")
    print("    ↓ _gql_generative_provider_key_to_semconv_provider()")
    print("  过滤: 'openai' (小写)")
    print("    ↓ 数据库查询: WHERE provider = 'openai'")
    print("  数据: records with provider='openai'")
    print("    ↓ _semconv_provider_to_gql_generative_provider_key()")
    print("  返回: OPENAI enum")
else:
    print("❌ 部分测试失败!")
