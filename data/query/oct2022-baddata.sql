create or replace table `g-data-gojek-id-mart.merchant_platform.dm_p_sanctioned_merchant_fake_tagging_bank_acc_oct` as
with sanction_all as(
select distinct
 target_id as saudagar_id,
 last_value(action_name) over(partition by target_id order by event_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_action,
--  last_value(execution_type) over(partition by target_id order by event_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_execution_type,
 last_value(rule_id) over(partition by target_id order by event_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_rule,
 last_value(datetime(event_timestamp,'Asia/Jakarta')) over(partition by target_id order by event_timestamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as last_event_timestamp
from
`data-gojek-id-standardized.mainstream.sanction_execution_log` a
where
 succes_flag
 and target_type = 'merchant'
 and lower(action_name) not like '%close%' and lower(action_name) not like '%open%' and lower(action_name) not like '%gofin%'
 and action_name not in ('DoNothing')
 and rule_id not like 'IBT-M52-018%' --Exclude suspension rule due to ATO/unauthorized add outlet activity, because it's not merchant's fault
 and rule_id not like '%M56%' --Exclude quality rule suspension
 and date(event_timestamp, 'Asia/Jakarta') >= '2019-01-01'
 and date(event_timestamp, 'Asia/Jakarta') >= date_trunc(date_sub(current_date('Asia/Jakarta'), interval 3 year), year)
 and (lower(a.action_name) like '%suspend%goresto' or lower(a.action_name) like '%suspend%payout')
 and target_id not like 'VN%'
)
-- get last action for merchant
, sanction_last as(
select
 saudagar_id,
 last_action,
 last_rule,
 last_event_timestamp as last_sanction_datetime,
--  b.*
--  case when last_action is not null then true else false end as suspend_flag,
--  last_event_timestamp
from
 sanction_all a
--  join KTP_info b on a.saudagar_id = b.outlet_id
where
 last_action in (
   'SuspendMerchantPayout',
   'SuspendMerchantGoResto'
 )
 and date(last_event_timestamp) >= '2022-10-01'
 and date(last_event_timestamp) <= '2022-10-31'
)
,data_summary AS (
  select distinct entity_id
     ,a.outlet_id
     ,case when a.outlet_id = c.saudagar_id then 1 else 0 end as flag_suspended
     ,AEAD.DECRYPT_STRING(b.outlet_ck, a.id_image_url,a.outlet_id) as id_image_url
     ,AEAD.DECRYPT_STRING(b.outlet_ck, a.id_name,a.outlet_id) as id_name
     ,AEAD.DECRYPT_STRING(b.outlet_ck, a.id_number,a.outlet_id) as id_number
     ,AEAD.DECRYPT_STRING(b.outlet_ck, a.bank_account_name, a.outlet_id) bank_acc_name
     ,AEAD.DECRYPT_STRING(b.outlet_ck, a.bank_account_number, a.outlet_id) bank_acc_no
     ,AEAD.DECRYPT_STRING(b.outlet_ck, a.outlet_photo_url,a.outlet_id) as outlet_photo_url
from `data-gojek-id-presentation.merchant_platform.dim_outlet` a, unnest(entity_tag_list) entity_id
left join `data-gojek-id-presentation.merchant_platform_outlet_keyset.cryptokey` b on a.outlet_id = b.outlet_id
left join (select
                saudagar_id,
                last_action,
                last_rule,
                last_event_timestamp as last_sanction_datetime,
                from
                sanction_all a
                where
                last_action in (
                  'SuspendMerchantPayout',
                  'SuspendMerchantGoResto'
                )
)c on a.outlet_id = c.saudagar_id
where true
)
,summary as (
select b.entity_id
      ,a.*
      ,b.outlet_photo_url
      ,b.bank_acc_name
      ,b.bank_acc_no
  from sanction_last a
  join data_summary b on a.saudagar_id = b.outlet_id
)
select a.*
      ,count(distinct b.entity_id) as cnt_diff_entity_shared_bank_acc
      ,string_agg(distinct b.entity_id,',') as diff_entity_id_shared_bank_acc
      ,count(distinct b.outlet_id) as cnt_diff_entity_outlet_shared_bank_acc
      ,string_agg(distinct b.outlet_id,',') as  diff_entity_outlet_id_shared_bank_acc
  from summary a
  left join data_summary b on a.bank_acc_no = b.bank_acc_no
                          and a.entity_id <> b.entity_id
                          and b.flag_suspended = 1
 group by 1,2,3,4,5,6,7,8
;
select a.*
      ,b.gofood_id
      ,count(c.order_no) as cnt_all_order
      ,sum(c.actual_gmv_amount) as sum_gmv_all_order
      ,count(case when c.status_id = '3' then c.order_no end) as cnt_co
      ,sum(case when c.status_id = '3' then c.actual_gmv_amount else 0 end) as sum_gmv_co
      ,count(case when c.status_id in ('2','7') and c.driver_order_placed_timestamp is null  and
(c.cancel_group_name is NULL or c.cancel_group_name = 'PORTAL' or c.cancel_group_name = 'SYSTEM') and
(c.cancel_reason_in_en is NULL or lower(c.cancel_reason_in_en) not like "%portal merchant%") then c.order_no end) as cnt_maf_defect_order
      ,sum(case when c.status_id in ('2','7') and c.driver_order_placed_timestamp is null  and
(c.cancel_group_name is NULL or c.cancel_group_name = 'PORTAL' or c.cancel_group_name = 'SYSTEM') and
(c.cancel_reason_in_en is NULL or lower(c.cancel_reason_in_en) not like "%portal merchant%") then c.actual_gmv_amount else 0 end) as sum_gmv_maf_defect_order
      ,count(case when c.status_id = '2' and c.driver_order_placed_timestamp is not null then c.order_no end) as cnt_cadf 
      ,sum(case when c.status_id = '2' and c.driver_order_placed_timestamp is not null then c.actual_gmv_amount else 0 end) as sum_gmv_cadf
  from `g-data-gojek-id-mart.merchant_platform.dm_p_sanctioned_merchant_fake_tagging_bank_acc_oct` a
  left join (select b.outlet_id
              ,b.gofood_id
              ,b.jakarta_gofood_acquired_date
          from `data-gojek-id-mart.merchant_platform.detail_master_merchant_universe` b
        where true
          and b.jakarta_data_date >= date_sub(current_date(),interval 3 day)
        group by 1,2,3
       )b on a.saudagar_id = b.outlet_id
  left join (select c.restaurant_id
                   ,c.order_no
                   ,c.actual_gmv_amount
                   ,c.status_id
                   ,c.driver_order_placed_timestamp
                   ,c.cancel_group_name
                   ,c.cancel_reason_in_en
                   ,c.created_timestamp
              from `data-gojek-id-mart.gofood.detail_gofood_booking` c
             where true
               and date(c.created_timestamp,'Asia/Jakarta') >= '2022-10-01'
               and date(c.created_timestamp,'Asia/Jakarta') <= '2022-10-31'
             group by 1,2,3,4,5,6,7,8
            )c on b.gofood_id = c.restaurant_id
              and date_trunc(date(c.created_timestamp,'Asia/Jakarta'),month) = date_trunc(a.last_sanction_datetime,month)
 group by 1,2,3,4,5,6,7,8,9,10,11,12,13