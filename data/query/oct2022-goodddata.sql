WITH menu_to_outlet_mapping AS 
(
  select 
r.menu_group_sk
,STRING_AGG(DISTINCT r.outlet_id ORDER BY r.outlet_id) AS outlet_id
from `data-gojek-id-mart.gofood.summary_content_moderation_ticket_evaluation_daily` ticket
LEFT JOIN `data-gojek-id-presentation.gofood.dim_restaurant`  r on ticket.menu_group_sk=r.menu_group_sk
AND DATE(r.effective_timestamp, 'Asia/Jakarta')<='2022-10-11'
AND (DATE(r.expired_timestamp,"Asia/Jakarta")> '2022-10-11' or r.expired_timestamp IS NULL)
WHERE TRUE
AND ticket.jakarta_data_date = '2022-10-11'
AND ticket.country_code = 'ID'
AND ticket.menu_group_sk IS NOT NULL
AND r.outlet_id IS NOT NULL
group by 1
order by menu_group_sk
),

raw AS 
(SELECT
  ticket.*,
  dr.outlet_id,
  REPLACE(JSON_EXTRACT(menu_name_response_in_json, "$.evaluation_result.name.decision_reason"),'"','') AS menu_name_decision_reason,
  REPLACE(JSON_EXTRACT(item_text_response_in_json, "$.evaluation_result.name.decision_reason"),'"','') AS item_name_decision_reason,
  REPLACE(JSON_EXTRACT(item_text_response_in_json, "$.evaluation_result.description.decision_reason"),'"','') AS item_description_decision_reason,
  ROW_NUMBER() OVER(PARTITION BY ticket_uuid ORDER BY ticket.updated_timestamp DESC) AS row_num
FROM `data-gojek-id-mart.gofood.summary_content_moderation_ticket_evaluation_daily` ticket
LEFT JOIN menu_to_outlet_mapping dr ON ticket.menu_group_sk = dr.menu_group_sk 
WHERE jakarta_data_date = '2022-10-11'
  AND ticket.country_code = 'ID'
)

select * except (rnk) from(

SELECT
  jakarta_data_date,
  created_timestamp,
  evaluation_event_timestamp as evaluation_timestamp,
  'content_management' as ticket_type_name,
  moderation_ticket_type as entity_type_name,
  concat(
    concat(
      concat(case when moderation_ticket_type = 'menu_group_menu' then 'content_menu_'
                  when moderation_ticket_type = 'menu_group_menu_item' then 'content_item_' end
            , CAST(ticket_uuid as STRING))
          , '_')
        , CAST(FORMAT_TIMESTAMP('%Y%m%d%H%M%S', evaluation_event_timestamp, 'Asia/Jakarta') as string) ) as request_uuid,
  ticket_uuid,
  NULL as user_id,
  NULL as onboarding_id,
  NULL as change_request_id,
  outlet_id,
  menu_group_uuid,
  action_type,
  old_menu_name,
  submitted_menu_name,
  old_item_name,
  submitted_item_name,
  old_item_description,
  submitted_item_description,
  NULL as old_user_name,
  NULL as submitted_user_name,
  NULL as old_outlet_name,
  NULL as submitted_outlet_name,
  NULL as old_outlet_address,
  NULL as submitted_outlet_address,
  NULL as data_detail_in_json,
  struct(struct(menu_name_decision_name, menu_name_rejection_reason, menu_name_rejection_reason_uuid, menu_name_decision_reason) as menu_name,
         struct(item_name_decision_name, item_name_rejection_reason, item_name_rejection_reason_uuid, item_name_decision_reason) as item_name,
         struct(item_description_decision_name, item_description_rejection_reason, item_description_rejection_reason_uuid, item_description_decision_reason) as item_description,
         resolved_by_name)
         as model_detail_in_json,
  case when menu_name_decision_name in ('Challenge', 'Reject')
            or item_name_decision_name in ('Challenge', 'Reject')
            or item_description_decision_name in ('Challenge', 'Reject')
       then 'Challenge' else 'Pass' end as model_decision_name
  , row_number() over (partition by ticket_uuid, evaluation_event_timestamp) as rnk
  , 'False' as evaluated_by_ioi_flag
  , NULL as ioi_evaluated_timestamp
  , NULL as response_payload_in_json
  , NULL as outlet_name_decision_name	
  , NULL as outlet_address_decision_name	
  , NULL as outlet_name_actual_decision_name	
  , NULL as outlet_address_actual_decision_name	
  , NULL as outlet_name_decision_reason	
  , NULL as outlet_address_decision_reason
FROM raw
WHERE (
  moderation_ticket_type='menu_group_menu' AND
    (menu_name_decision_name = 'Reject'
      OR menu_name_rejection_reason_uuid IN ('938a58fb-9248-45d2-991b-8a04fe6142c2')
    )
) OR (
  moderation_ticket_type='menu_group_menu_item' AND (
    (item_name_decision_name = 'Reject' AND
      item_name_decision_reason IN ('item-stopword-punctuation',
                                    'item-pure-numeric',
                                    'item-scam-keyword',
                                    'item-bank-card-digit')
    ) OR (
    item_description_decision_name = 'Reject' AND
      item_description_decision_reason IN ('description-stopword-punctuation',
                                          'description-scam-keyword')
    )
  )
)

) where rnk = 1
order by 2
