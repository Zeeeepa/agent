from __future__ import annotations

class JinxError(Exception):
    pass

class ValidationError(JinxError):
    pass

class ImportSynthesisError(JinxError):
    pass

class RequirementsUpdateError(JinxError):
    pass

class VerificationError(JinxError):
    pass
