import os


from absl import flags


class ConfigParser:
    """
    配置优先级 代码默认配置 < 环境变量 < flags
    """

    def __init__(self, default_env_true_value="1", default_env_false_value="0") -> None:
        self.default_env_true_value = default_env_true_value
        self.default_env_false_value = default_env_false_value

    def _get_flag_name(self, prefix, key):
        return f"{prefix}_{key}"

    def _register_flag(self, flag_name, default_value, help_msg):
        if type(default_value) == str:
            flags.DEFINE_string(flag_name, default_value, help_msg)
        elif type(default_value) == int:
            flags.DEFINE_integer(flag_name, default_value, help_msg)
        elif type(default_value) == bool:
            flags.DEFINE_boolean(flag_name, default_value, help_msg)
        else:
            raise Exception(
                f"Unknown value type: {type(default_value)} for {default_value}"
            )

    def _get_env_bool_val(self, env_name, default_config, default_value):
        # type(env_val) == str or None
        env_val = os.getenv(env_name)

        # type(value) == str
        true_value = default_config.get("env_true_value", self.default_env_true_value)
        false_value = default_config.get(
            "env_false_value", self.default_env_false_value
        )

        if type(true_value) is not str or type(false_value) is not str:
            raise Exception(
                f"value({true_value},{false_value}) for {env_name} is not str"
            )

        # overwrite with env_val
        if env_val == true_value:
            return True
        elif env_val == false_value:
            return False
        else:
            return default_value

    def _get_env_val(self, env_name, default_config, default_value):
        """
        get val from env
        Return value parsed from env which has the same type as default_value or default_value
        """
        value_type = type(default_value)
        if value_type is bool:
            return self._get_env_bool_val(env_name, default_config, default_value)
        else:
            return value_type(os.getenv(env_name, default_value))

    def register_config_to_flags(self, prefix, config_definition, default_config=None):
        default_config = default_config or {}

        for k, v in config_definition.items():
            assert type(v) == dict

            # 优先级: v["value"] < default_config[k] < env_alias < env < flags
            default_value = default_config.get(k, v.get("value"))
            if default_value is None:
                raise Exception("default_value is None")

            env_names = v.get("env_alias", []) + [self._get_env_name(prefix, k)]

            for env_name in env_names:
                default_value = self._get_env_val(
                    env_name, default_config, default_value
                )

            help_msg = v.get("help", "")
            flag_name = self._get_flag_name(prefix, k)
            self._register_flag(
                flag_name, default_value, help_msg + f"(ENV: {env_names})"
            )

            for alia in v.get("flag_alias", []):
                flags.DEFINE_alias(alia, original_name=flag_name)
            #    self._register_flag(flag_name, default_value, help_msg)

    def _get_config_from_flags(self, prefix, key):
        flag_name = self._get_flag_name(prefix, key)
        return getattr(flags.FLAGS, flag_name)

    def _get_env_name(self, prefix, key):
        return f"{prefix}_{key}".upper()

    def parse(self, prefix, config_definition):
        ret = {}
        for k, v in config_definition.items():
            assert type(v) == dict

            flag_name = self._get_flag_name(prefix, k)
            ret[k] = getattr(flags.FLAGS, flag_name)
        return ret


if __name__ == "__main__":

    config_definition = {
        "config_1": {
            "value": "config_1_value",  # 必须, 既作为默认值的存在, 也需要使用它的类型
            "help": "123",  # 可选
            "env_alias": ["ENV"],  # 可选
            "flag_alias": ["flags"],  # 可选
        },
        "config_2": {
            "value": "config_2_value",  # 必须, 既作为默认值的存在, 也需要使用它的类型
        },
    }

    # 设置config的默认值, 可以不是全量
    config_default = {
        "config_1": "config_1_value",
        "config_2": "config_2_value",
    }

    parser = ConfigParser()
    parser.register_config_to_flags("a", config_definition, config_default)

    def main(_):
        config = parser.parse("a", config_definition)
        print(config)

    from absl import app

    app.run(main)
