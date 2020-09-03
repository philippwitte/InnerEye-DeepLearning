

def add_logger_args(parser):
    group = parser.add_argument_group("Logging arguments")
    group.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Sets the logging level.")
