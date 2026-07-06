# NOTE: requires system imagemagick install for `magick` command (`convert` is deprecated)

require 'fileutils'
require 'json'

INPUT_FILE = 'data/gardiner_hieroglyphs_with_unicode_hex.json'
OUTPUT_DIR = 'data/utf-pngs'
FONT_PATH  = 'lib/font/NotoSansEgyptianHieroglyphs-Regular.ttf'
IMG_SIZE   = 400

FileUtils.mkdir_p OUTPUT_DIR

# read JSON as array of (glyph) hashes
glyphs = JSON.parse(File.read(INPUT_FILE))

glyphs.each do |glyph|
  hieroglyph    = glyph['hieroglyph']
  unicode_hex   = glyph['unicode_hex']
  gardiner_num  = glyph['gardiner_num']
  png_path      = File.join(OUTPUT_DIR, "#{gardiner_num}.png")
  
  next unless !unicode_hex.empty? && !hieroglyph.empty?

  puts "Generating PNG for '#{hieroglyph}' at #{png_path}"
  # takes the UTF glyph directly as label opt, generates & saves image representation to png_path
  # uses ttf font path directly from repo (most others do not contain egyptian glyphs in their charsets)
  # direct system/sh call instead of using rmagick gem (i'm lazy)
  # image size (pixels) defined above
  system `magick -pointsize #{IMG_SIZE} -font #{FONT_PATH} label:#{hieroglyph} #{png_path}`
end
