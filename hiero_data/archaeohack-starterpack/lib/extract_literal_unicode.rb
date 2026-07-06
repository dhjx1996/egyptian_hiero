require 'csv'
require 'json'

INPUT_FILE = 'data/gardiner_hieroglyphs.csv'
OUTPUT_FILE = 'data/gardiner_hieroglyphs_with_unicode_hex.json'

# read CSV as array of hashes
glyphs = CSV.parse(File.read(INPUT_FILE), headers: true).map(&:to_h)
# transform in place, add new `unicode_hex` code directly to hash w/ map!
glyphs.map! do |glyph|
  # extract ASCII literal unicode from hieroglyph field 
  # use error msg itself to get that info if necessary (char encoding is a weeeeeird thing)
  begin
    unicode = glyph['hieroglyph'].encode("ASCII") # try to encode it to ASCII
  rescue Encoding::UndefinedConversionError # but if that fails
    unicode = $!.error_char.dump # capture a dump, mapped to the source character
  end
  glyph['unicode_hex'] = unicode.gsub /[\D]/, '' # store it as clean digits for hex, remove extra non-digit chars
  glyph['is_priority'] = glyph['is_priority'].to_s.downcase == "true" # convert to boolean
  glyph # return updated hash
end

# write results to nicely-formatted JSON file
File.write OUTPUT_FILE, JSON.pretty_generate(glyphs)
