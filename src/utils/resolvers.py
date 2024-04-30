from omegaconf import OmegaConf


def register_resolvers():
    OmegaConf.register_new_resolver("as_tuple", resolve_tuple)
    OmegaConf.register_new_resolver("sanitize_str", sanitize_str)


def resolve_tuple(args):
    return tuple(args)


def sanitize_str(s):
    return str.replace(s, "/", "_").replace(r'"', "")
