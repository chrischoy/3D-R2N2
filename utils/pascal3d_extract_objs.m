%
% Extract object segments in PASCAL3D+ dataset for 3D-R2N2
% Copyright (C) 2016  Xingyou Chen <niatlantice@gmail.com>
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

pad_ratio= 0.1;
annos = textread('pascal3d.list', '%s');
n_annos = size(annos, 1);
for anno = 1:n_annos
	disp(anno);
	load(annos{anno});	% loads record

	if (strcmp(record.database, 'ImageNet'))
		img_path = sprintf('../Images/%s.JPEG', annos{anno}(1:end-4));
	else
		img_path = sprintf('../Images/%s.jpg', annos{anno}(1:end-4));
	end

	img = imread(img_path);
	[img_h, img_w, channel] = size(img);
	n_objs = size(record.objects, 2);

	for idx = 1:n_objs
		cur_obj = record.objects(idx);
		anno_info = {};
		bbox = round(cur_obj.bbox);
		if (bbox(1) < 1)
			bbox(1) = 1;
		end
		if (bbox(2) < 1)
			bbox(2) = 1;
		end
		if (bbox(3) > img_w)
			bbox(3) = img_w;
		end
		if (bbox(4) > img_h)
			bbox(4) = img_h;
		end
		% Ignore bad quality images
		if (cur_obj.occluded > 0) || (cur_obj.truncated > 0) || ((bbox(3) - bbox(1)) < 120) || ((bbox(4) - bbox(2)) < 120)
			continue;
		end

		% Padding stage 1: expand on original image
		expand = max(bbox(3) - bbox(1), bbox(4) - bbox(2));
		expand = round(expand * (1 + pad_ratio));
		ebbox(1) = round((bbox(1) + bbox(3) - expand) / 2);
		ebbox(2) = round((bbox(2) + bbox(4) - expand) / 2);
		ebbox(3) = bbox(1) + expand;
		ebbox(4) = bbox(2) + expand;
		if (ebbox(1) < 1)
			ebbox(1) = 1;
		end
		if (ebbox(2) < 1)
			ebbox(2) = 1;
		end
		if (ebbox(3) > img_w)
			ebbox(3) = img_w;
		end
		if (ebbox(4) > img_h)
			ebbox(4) = img_h;
		end 

		save_path = sprintf('../anno_sel/%s_%d.jpg', annos{anno}(1:end-4), idx);
		anno_path = sprintf('../anno_sel/%s_%d.mat', annos{anno}(1:end-4), idx);

		anno_info.db = record.database;
		anno_info.class = cur_obj.class;
		anno_info.difficult = cur_obj.difficult;
		anno_info.cad_index = cur_obj.cad_index;
		anno_info.viewpoint = cur_obj.viewpoint;
		anno_info.anchors = cur_obj.anchors;

		roi = img(ebbox(2):ebbox(4), ebbox(1):ebbox(3), :);
		[h, w, channel] = size(roi);

		% Padding stage 2: filling missing pixels
		clear pad_img;
		if (w > h)
			pad_h = round((w - h)/2);
			pad_img((pad_h + 1):(pad_h + h), :, :) = roi;
			pad_img(1:pad_h, :, :) = repmat(roi(1, :, :), pad_h, 1);
			if ((w - h) > 1)
				pad_img((pad_h + h + 1):w, :, :) = repmat(roi(h, :, :), (w - h - pad_h), 1, 1);
			end
		elseif (h > w)		% case (w==h) can be safely ignored
			pad_w = round((h - w) / 2);
			pad_img(:, (pad_w + 1):(pad_w + w), :) = roi;
			pad_img(:, 1:pad_w, :) = repmat(roi(:, 1, :), 1, pad_w);
			if ((h - w) > 1)
				pad_img(:, (pad_w + w + 1):h, :) = repmat(roi(:, w, :), 1, (h - w - pad_w), 1);
			end
		else
			pad_img = roi;
		end
		pad_img = imresize(pad_img, [127, 127]);
		imwrite(pad_img, save_path);
		save(anno_path, 'anno_info');
	end
end
