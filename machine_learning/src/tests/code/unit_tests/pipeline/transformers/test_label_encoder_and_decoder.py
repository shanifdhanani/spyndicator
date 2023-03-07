from ml_core.src.main.code.pipeline.transformers.label_encoder_and_decoder import LabelEncoderAndDecoder


class TestLabelEncoderAndDecoder:
    def setup(self):
        self.label_encoder_and_decoder = LabelEncoderAndDecoder()

    def test_encoder_encoder_and_decodes_correctly(self):
        labels = ['hello', 'whiskey', 'tango', 'twitter', 'apple']
        encoded_labels = self.label_encoder_and_decoder.transform(labels)
        decoded_labels = self.label_encoder_and_decoder.decode(encoded_labels)

        assert list(labels) == list(decoded_labels)

    def test_new_values_get_encoded_correctly(self):
        labels = ['hello', 'whiskey', 'tango', 'twitter', 'apple']
        encoded_labels = self.label_encoder_and_decoder.transform(labels)
        encoded_label_for_new_value = self.label_encoder_and_decoder.transform([LabelEncoderAndDecoder.OtherToken])[0]
        new_value_labels = ['value-not-present', 'value-not-present-2']
        new_value_encoded_labels = self.label_encoder_and_decoder.transform(new_value_labels)
        for encoded_label in new_value_encoded_labels:
            assert encoded_label == encoded_label_for_new_value
