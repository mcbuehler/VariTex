from varitex.options.base_options import BaseOptions


class EvalOptions(BaseOptions):

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--path_opt', default=None, help='json with options.')
        return parser
