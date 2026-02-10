from app.mentor_net.noisy_scaner import NoiseScanner


class MLNoiseService:
    def __init__(self, trainer):
        self.scanner = NoiseScanner(trainer)

    def run_noise_scan(self, loader, original_dataframe):
        full_df = self.scanner.scan_dataset(loader, original_dataframe)
        suspicious_df, stats_context = self.scanner.prepare_for_agent(full_df)

        return {
            "suspicious_df": suspicious_df,
            "stats_context": stats_context,
            "full_df": full_df,
        }
