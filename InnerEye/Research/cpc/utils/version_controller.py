import git


class VersionController(git.Repo):
    def __init__(self, path=None):
        super(VersionController, self).__init__(
            path=path or __file__,
            search_parent_directories=True
        )

    @property
    def remote_url(self):
        return str(self.remotes[0].url)

    @property
    def current_commit(self):
        return str(next(self.iter_commits()))

    @property
    def branch(self):
        return str(self.active_branch)

    @property
    def user_name(self):
        return str(self.git.config("user.name"))
