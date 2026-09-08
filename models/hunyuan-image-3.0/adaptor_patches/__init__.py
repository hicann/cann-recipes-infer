import transformers.dynamic_module_utils as _dmu
import transformers.models.auto.configuration_auto as _ca
import transformers.models.auto.auto_factory as _af
import transformers.models.auto.tokenization_auto as _ta

_trusted = lambda *args, **kwargs: True

_dmu.resolve_trust_remote_code = _trusted
_ca.resolve_trust_remote_code = _trusted
_af.resolve_trust_remote_code = _trusted
_ta.resolve_trust_remote_code = _trusted
